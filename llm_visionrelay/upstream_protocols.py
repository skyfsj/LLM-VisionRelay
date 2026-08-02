"""Upstream protocol adapters.

The upstream text model can speak OpenAI Chat Completions (default), Anthropic
Messages, or OpenAI Responses. The middleware always works in the chat format
internally; :class:`UpstreamAdapter` renders chat payloads into the configured
upstream protocol and parses responses / SSE streams back into chat format.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import httpx

from llm_visionrelay.errors import UpstreamNonJsonError
from llm_visionrelay.headers import RequestConfig
from llm_visionrelay.protocols import (
    PROTOCOL_ANTHROPIC,
    PROTOCOL_RESPONSES,
)
from llm_visionrelay.upstream import UpstreamClient, UpstreamResult

DEFAULT_MAX_TOKENS = 4096

_FINISH_FROM_STOP: dict[str, str] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "refusal": "content_filter",
}


def _safe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": text}


def upstream_chat_endpoint(base_url: str, protocol: str) -> str:
    base = base_url.rstrip("/")
    if protocol == PROTOCOL_ANTHROPIC:
        return base + "/v1/messages"
    if protocol == PROTOCOL_RESPONSES:
        return base + "/responses"
    return base + "/chat/completions"


def upstream_models_endpoint(base_url: str, protocol: str) -> str:
    base = base_url.rstrip("/")
    if protocol == PROTOCOL_ANTHROPIC:
        return base + "/v1/models"
    return base + "/models"


# ---------------------------------------------------------------------------
# Render: chat payload -> upstream protocol request body
# ---------------------------------------------------------------------------
def _chat_message_to_anthropic(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    content = message.get("content")
    if role == "tool":
        tool_call_id = message.get("tool_call_id") or ""
        result = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": result}],
        }

    blocks: list[dict[str, Any]] = []
    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                blocks.append({"type": "text", "text": block.get("text") or ""})
    if role == "assistant":
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                    "name": fn.get("name") or "",
                    "input": _safe_json(fn.get("arguments") or ""),
                }
            )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return {"role": role, "content": blocks}


def _chat_tool_to_anthropic(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool.get("function") or {}
    return {
        "name": fn.get("name") or tool.get("name") or "",
        "description": fn.get("description") or tool.get("description") or "",
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def render_chat_to_anthropic(payload: dict[str, Any]) -> dict[str, Any]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.append(
                    "".join(
                        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                    )
                )
            continue
        messages.append(_chat_message_to_anthropic(message))

    body: dict[str, Any] = {
        "model": payload.get("model"),
        "max_tokens": payload.get("max_completion_tokens") or payload.get("max_tokens") or DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    if system_parts:
        body["system"] = system_parts[0] if len(system_parts) == 1 else system_parts
    for key in ("temperature", "top_p", "stop"):
        if payload.get(key) is not None:
            body[key] = payload[key]
    effort = payload.get("reasoning_effort")
    if isinstance(payload.get("thinking"), dict):
        body["thinking"] = deepcopy(payload["thinking"])
    elif isinstance(effort, str) and effort:
        body["thinking"] = {"type": "enabled", "budget_tokens": _effort_to_budget(effort)}
    if payload.get("tool_choice") is not None:
        body["tool_choice"] = _chat_tool_choice_to_anthropic(payload["tool_choice"])
    if payload.get("metadata") is not None:
        body["metadata"] = deepcopy(payload["metadata"])
    if payload.get("stream"):
        body["stream"] = True
    tools = payload.get("tools")
    if tools:
        body["tools"] = [_chat_tool_to_anthropic(t) for t in tools if isinstance(t, dict)]
    return body


def _effort_to_budget(effort: str) -> int:
    return {
        "low": 2048,
        "medium": 8192,
        "high": 16384,
        "xhigh": 32768,
    }.get(str(effort).lower(), 8192)


def _chat_tool_choice_to_anthropic(tool_choice: Any) -> dict[str, Any]:
    if isinstance(tool_choice, str):
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return {"type": "none"}
        return {"type": "auto"}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fn = tool_choice.get("function") or {}
        return {"type": "tool", "name": fn.get("name", "")}
    return {"type": "auto"}


def _chat_tool_choice_to_responses(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fn = tool_choice.get("function") or {}
        return {"type": "function", "name": fn.get("name", "")}
    return "auto"


def _chat_message_to_responses_item(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    if role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.get("tool_call_id") or "",
                "output": message.get("content")
                if isinstance(message.get("content"), str)
                else json.dumps(message.get("content"), ensure_ascii=False),
            }
        ]
    content = message.get("content")
    blocks: list[dict[str, Any]] = []
    if isinstance(content, str):
        if content:
            blocks.append({"type": "input_text", "text": content})
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                blocks.append({"type": "input_text", "text": block.get("text") or ""})
    if role == "assistant" and message.get("tool_calls"):
        items: list[dict[str, Any]] = []
        for tc in message["tool_calls"]:
            fn = tc.get("function") or {}
            items.append(
                {
                    "type": "function_call",
                    "call_id": tc.get("id") or "",
                    "name": fn.get("name") or "",
                    "arguments": fn.get("arguments") or "{}",
                }
            )
        return items
    return [{"role": role or "user", "content": blocks}]


def render_chat_to_responses(payload: dict[str, Any]) -> dict[str, Any]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                instructions.append(content)
            elif isinstance(content, list):
                instructions.append(
                    "".join(
                        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                    )
                )
            continue
        input_items.extend(_chat_message_to_responses_item(message))

    body: dict[str, Any] = {"model": payload.get("model"), "input": input_items}
    if instructions:
        body["instructions"] = "\n".join(instructions)
    for key in ("temperature", "top_p"):
        if payload.get(key) is not None:
            body[key] = payload[key]
    if payload.get("max_completion_tokens") is not None:
        body["max_output_tokens"] = payload["max_completion_tokens"]
    elif payload.get("max_tokens") is not None:
        body["max_output_tokens"] = payload["max_tokens"]
    elif payload.get("max_output_tokens") is not None:
        body["max_output_tokens"] = payload["max_output_tokens"]
    effort = payload.get("reasoning_effort")
    if isinstance(effort, str) and effort:
        body["reasoning"] = {"effort": effort}
    elif isinstance(payload.get("reasoning"), dict):
        body["reasoning"] = deepcopy(payload["reasoning"])
    if payload.get("tool_choice") is not None:
        body["tool_choice"] = _chat_tool_choice_to_responses(payload["tool_choice"])
    for key in ("parallel_tool_calls", "store", "user", "metadata", "truncation"):
        if payload.get(key) is not None:
            body[key] = deepcopy(payload[key])
    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        rtype = response_format.get("type")
        if rtype in ("json_object", "json_schema"):
            text: dict[str, Any] = {"format": "json_schema"}
            if isinstance(response_format.get("json_schema"), dict):
                text["schema"] = deepcopy(response_format["json_schema"])
            body["text"] = text
        elif rtype == "text":
            body["text"] = {"format": "plain_text"}
    if payload.get("stream"):
        body["stream"] = True
    tools = payload.get("tools")
    if tools:
        converted = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            name = fn.get("name") or tool.get("name")
            if name:
                converted.append(
                    {
                        "type": "function",
                        "name": name,
                        "description": fn.get("description") or tool.get("description") or "",
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
        body["tools"] = converted
    return body


# ---------------------------------------------------------------------------
# Parse: upstream protocol response -> chat response
# ---------------------------------------------------------------------------
def _normalize_anthropic_error(err: dict[str, Any]) -> dict[str, Any]:
    return {
        "message": err.get("message") or "anthropic upstream error",
        "type": "anthropic_error",
        "param": None,
        "code": err.get("type") or "api_error",
    }


def parse_anthropic_to_chat(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("error") is not None:
        inner = data["error"] if isinstance(data["error"], dict) else {"message": str(data["error"])}
        return {"error": _normalize_anthropic_error(inner)}

    text: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text.append(block.get("text") or "")
        elif btype == "thinking":
            reasoning.append(block.get("thinking") or "")
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )

    message: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = data.get("usage") or {}
    return {
        "id": data.get("id") or f"chatcmpl_{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _FINISH_FROM_STOP.get(data.get("stop_reason"), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        },
    }


def parse_responses_to_chat(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("error") is not None:
        return {
            "error": data["error"] if isinstance(data["error"], dict) else {"message": str(data["error"])}
        }

    text: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text.append(part.get("text") or "")
        elif itype == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {"name": item.get("name") or "", "arguments": item.get("arguments") or ""},
                }
            )
        elif itype == "reasoning":
            for summary in item.get("summary") or []:
                if isinstance(summary, dict):
                    reasoning.append(summary.get("text") or "")

    message: dict[str, Any] = {"role": "assistant", "content": "".join(text) or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = data.get("usage") or {}
    return {
        "id": data.get("id") or f"chatcmpl_{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": data.get("created_at") or int(time.time()),
        "model": data.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Streaming: upstream SSE events -> chat chunks
# ---------------------------------------------------------------------------
def _chunk(rid: str, model: str, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class AnthropicStreamToChat:
    def __init__(self) -> None:
        self.id = ""
        self.model = ""
        self.block_kinds: dict[int, str] = {}
        self.tool_index: dict[int, int] = {}
        self.next_tool = 0
        self.role_sent = False
        self.finish_sent = False

    def _role(self) -> list[dict[str, Any]]:
        if self.role_sent:
            return []
        self.role_sent = True
        return [_chunk(self.id, self.model, {"role": "assistant", "content": ""})]

    def handle(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        etype = data.get("type")
        if etype == "message_start":
            msg = data.get("message") or {}
            self.id = msg.get("id") or self.id
            self.model = msg.get("model") or self.model
            chunks.extend(self._role())
        elif etype == "content_block_start":
            index = data.get("index", 0)
            block = data.get("content_block") or {}
            kind = block.get("type") or "text"
            self.block_kinds[index] = kind
            if kind == "tool_use":
                self.tool_index[index] = self.next_tool
                self.next_tool += 1
                chunks.extend(self._role())
                chunks.append(
                    _chunk(
                        self.id,
                        self.model,
                        {
                            "tool_calls": [
                                {
                                    "index": self.tool_index[index],
                                    "id": block.get("id"),
                                    "type": "function",
                                    "function": {"name": block.get("name") or "", "arguments": ""},
                                }
                            ]
                        },
                    )
                )
        elif etype == "content_block_delta":
            index = data.get("index", 0)
            delta = data.get("delta") or {}
            dtype = delta.get("type")
            chunks.extend(self._role())
            if dtype == "text_delta":
                chunks.append(_chunk(self.id, self.model, {"content": delta.get("text") or ""}))
            elif dtype == "input_json_delta":
                tool_index = self.tool_index.get(index)
                if tool_index is not None:
                    chunks.append(
                        _chunk(
                            self.id,
                            self.model,
                            {
                                "tool_calls": [
                                    {
                                        "index": tool_index,
                                        "function": {"arguments": delta.get("partial_json") or ""},
                                    }
                                ]
                            },
                        )
                    )
            elif dtype == "thinking_delta":
                chunks.append(_chunk(self.id, self.model, {"reasoning_content": delta.get("thinking") or ""}))
        elif etype == "message_delta":
            stop = (data.get("delta") or {}).get("stop_reason")
            finish = _FINISH_FROM_STOP.get(stop) if stop else None
            if finish and not self.finish_sent:
                self.finish_sent = True
                chunks.append(_chunk(self.id, self.model, {}, finish))
        return chunks

    def finish(self) -> list[dict[str, Any]]:
        if self.finish_sent:
            return []
        self.finish_sent = True
        return [_chunk(self.id, self.model, {}, "stop")]


class ResponsesStreamToChat:
    def __init__(self) -> None:
        self.id = ""
        self.model = ""
        self.tool_index: dict[int, int] = {}
        self.next_tool = 0
        self.role_sent = False
        self.finish_sent = False

    def _role(self) -> list[dict[str, Any]]:
        if self.role_sent:
            return []
        self.role_sent = True
        return [_chunk(self.id, self.model, {"role": "assistant", "content": ""})]

    def handle(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        etype = data.get("type")
        if etype == "response.created":
            resp = data.get("response") or {}
            self.id = resp.get("id") or self.id
            self.model = resp.get("model") or self.model
            chunks.extend(self._role())
        elif etype == "response.output_item.added":
            item = data.get("item") or {}
            output_index = data.get("output_index", 0)
            if item.get("type") == "function_call":
                self.tool_index[output_index] = self.next_tool
                self.next_tool += 1
                chunks.extend(self._role())
                chunks.append(
                    _chunk(
                        self.id,
                        self.model,
                        {
                            "tool_calls": [
                                {
                                    "index": self.tool_index[output_index],
                                    "id": item.get("call_id") or item.get("id"),
                                    "type": "function",
                                    "function": {"name": item.get("name") or "", "arguments": ""},
                                }
                            ]
                        },
                    )
                )
        elif etype == "response.output_text.delta":
            chunks.extend(self._role())
            chunks.append(_chunk(self.id, self.model, {"content": data.get("delta") or ""}))
        elif etype == "response.function_call_arguments.delta":
            output_index = data.get("output_index", 0)
            tool_index = self.tool_index.get(output_index)
            if tool_index is not None:
                chunks.append(
                    _chunk(
                        self.id,
                        self.model,
                        {
                            "tool_calls": [
                                {"index": tool_index, "function": {"arguments": data.get("delta") or ""}}
                            ]
                        },
                    )
                )
        elif etype in ("response.completed", "response.in_progress"):
            if not self.finish_sent:
                self.finish_sent = True
                finish = "tool_calls" if self.next_tool else "stop"
                chunks.append(_chunk(self.id, self.model, {}, finish))
        return chunks

    def finish(self) -> list[dict[str, Any]]:
        if self.finish_sent:
            return []
        self.finish_sent = True
        finish = "tool_calls" if self.next_tool else "stop"
        return [_chunk(self.id, self.model, {}, finish)]


def _new_stream_state(protocol: str) -> Any:
    if protocol == PROTOCOL_ANTHROPIC:
        return AnthropicStreamToChat()
    if protocol == PROTOCOL_RESPONSES:
        return ResponsesStreamToChat()
    return None


class UpstreamAdapter:
    """Render/parse a chat-format payload against a specific upstream protocol."""

    def __init__(self, protocol: str, client: UpstreamClient) -> None:
        self.protocol = protocol
        self.client = client

    def endpoint(self, base_url: str) -> str:
        return upstream_chat_endpoint(base_url, self.protocol)

    def render(self, chat_payload: dict[str, Any]) -> dict[str, Any]:
        if self.protocol == PROTOCOL_ANTHROPIC:
            return render_chat_to_anthropic(chat_payload)
        if self.protocol == PROTOCOL_RESPONSES:
            return render_chat_to_responses(chat_payload)
        return chat_payload

    def parse_json(self, resp: httpx.Response) -> dict[str, Any]:
        try:
            body = resp.json()
        except (ValueError, TypeError) as exc:
            raise UpstreamNonJsonError() from exc
        if self.protocol == PROTOCOL_ANTHROPIC:
            return parse_anthropic_to_chat(body)
        if self.protocol == PROTOCOL_RESPONSES:
            return parse_responses_to_chat(body)
        return body

    def _headers(self, cfg: RequestConfig) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if cfg.authorization:
            headers["Authorization"] = cfg.authorization
        return headers

    async def request_json(self, cfg: RequestConfig, chat_payload: dict[str, Any]) -> UpstreamResult:
        url = self.endpoint(cfg.upstream_base_url)
        content = json.dumps(self.render(chat_payload), ensure_ascii=False).encode()
        resp = await self.client.post_bytes(url, self._headers(cfg), content)
        return UpstreamResult(resp.status_code, self.parse_json(resp))

    async def stream_chunks(
        self, cfg: RequestConfig, chat_payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        rendered = dict(self.render(chat_payload))
        rendered["stream"] = True
        url = self.endpoint(cfg.upstream_base_url)
        content = json.dumps(rendered, ensure_ascii=False).encode()
        resp = await self.client.stream_bytes(url, self._headers(cfg), content)
        state = _new_stream_state(self.protocol)
        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if state is None:
                    yield obj
                else:
                    for chunk in state.handle(obj):
                        yield chunk
            if state is not None:
                for chunk in state.finish():
                    yield chunk
        finally:
            await resp.aclose()


def build_adapter(upstream: UpstreamClient, protocol: str) -> UpstreamAdapter:
    return UpstreamAdapter(protocol, upstream)
