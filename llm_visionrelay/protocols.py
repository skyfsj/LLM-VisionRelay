"""Protocol adapters for OpenAI Chat Completions, Anthropic Messages, and
OpenAI Responses API.

The middleware normalizes every client protocol into OpenAI Chat Completions
format (the canonical upstream representation), runs the vision pipeline and
tool loop in that format, then renders the upstream response back into the
client's protocol — including streaming SSE and external tool calls.

Protocol detection is automatic from the request path:

- ``POST /v1/chat/completions``  -> ``chat``
- ``POST /v1/messages``          -> ``anthropic``
- ``POST /v1/responses``         -> ``responses``
"""

from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from llm_visionrelay.errors import InvalidRequestBody

PROTOCOL_CHAT = "chat"
PROTOCOL_ANTHROPIC = "anthropic"
PROTOCOL_RESPONSES = "responses"


@dataclass
class NormalizedRequest:
    """A client request normalized to the internal (chat) representation."""

    protocol: str
    model: str | None
    stream: bool
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    base_body: dict[str, Any]


def protocol_from_path(path: str) -> str:
    if path.rstrip("/").endswith("/v1/messages"):
        return PROTOCOL_ANTHROPIC
    if path.rstrip("/").endswith("/v1/responses"):
        return PROTOCOL_RESPONSES
    return PROTOCOL_CHAT


def _safe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": text}


def _anthropic_tool_choice_to_chat(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        t = tool_choice.get("type")
        if t == "any":
            return "required"
        if t == "tool":
            return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return "auto"


def _responses_tool_choice_to_chat(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return "auto"


# ---------------------------------------------------------------------------
# Parsing: client protocol -> chat messages/tools/base_body
# ---------------------------------------------------------------------------
def parse_request(protocol: str, body: dict[str, Any], config: Any) -> NormalizedRequest:
    if protocol == PROTOCOL_CHAT:
        return _parse_chat(body)
    if protocol == PROTOCOL_ANTHROPIC:
        return _parse_anthropic(body)
    if protocol == PROTOCOL_RESPONSES:
        return _parse_responses(body)
    raise InvalidRequestBody(f"unsupported protocol {protocol!r}")


def _parse_chat(body: dict[str, Any]) -> NormalizedRequest:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise InvalidRequestBody("messages must be a list")
    tools = body.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise InvalidRequestBody("tools must be a list")
    base_body = deepcopy(body)
    base_body.pop("stream", None)
    return NormalizedRequest(
        protocol=PROTOCOL_CHAT,
        model=body.get("model"),
        stream=bool(body.get("stream", False)),
        messages=messages,
        tools=tools,
        base_body=base_body,
    )


def _anthropic_system_text(system: Any) -> list[str]:
    if isinstance(system, str):
        return [system]
    if isinstance(system, list):
        return [b.get("text") or "" for b in system if isinstance(b, dict) and b.get("type") == "text"]
    return []


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                parts.append(b.get("text") or "")
            elif b.get("type") == "image":
                parts.append("[image tool result]")
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _anthropic_image_to_chat_block(block: dict[str, Any]) -> dict[str, Any]:
    source = block.get("source") or {}
    stype = source.get("type")
    detail = block.get("detail") or "auto"
    if stype == "base64":
        media = source.get("media_type") or "image/png"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media};base64,{source.get('data', '')}", "detail": detail},
        }
    if stype == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", ""), "detail": detail}}
    raise InvalidRequestBody(f"unsupported anthropic image source type {stype!r}")


def _anthropic_message_to_chat(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    content = message.get("content")

    if role == "assistant":
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text") or "")
                elif btype == "thinking":
                    text_parts.append(block.get("thinking") or "")
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": block.get("name") or "",
                                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                            },
                        }
                    )
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return [msg]

    if role == "user":
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        if isinstance(content, list):
            blocks = [b for b in content if isinstance(b, dict)]
            if any(b.get("type") == "tool_result" for b in blocks):
                out: list[dict[str, Any]] = []
                user_parts: list[dict[str, Any]] = []
                for b in blocks:
                    btype = b.get("type")
                    if btype == "text":
                        user_parts.append({"type": "text", "text": b.get("text") or ""})
                    elif btype == "image":
                        user_parts.append(_anthropic_image_to_chat_block(b))
                if user_parts:
                    out.append({"role": "user", "content": user_parts})
                for b in blocks:
                    if b.get("type") == "tool_result":
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": b.get("tool_use_id") or "",
                                "content": _tool_result_text(b.get("content")),
                            }
                        )
                return out
            parts: list[dict[str, Any]] = []
            for b in blocks:
                btype = b.get("type")
                if btype == "text":
                    parts.append({"type": "text", "text": b.get("text") or ""})
                elif btype == "image":
                    parts.append(_anthropic_image_to_chat_block(b))
            return [{"role": "user", "content": parts}]
        return [{"role": "user", "content": ""}]

    if role in ("system", "developer"):
        if isinstance(content, str):
            return [{"role": "system", "content": content}]
        if isinstance(content, list):
            text = "".join(
                b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
            return [{"role": "system", "content": text}]
        return [{"role": "system", "content": ""}]

    return [
        {
            "role": role,
            "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
        }
    ]


def _anthropic_tools_to_chat(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description") or "",
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def _parse_anthropic(body: dict[str, Any]) -> NormalizedRequest:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise InvalidRequestBody("messages must be a list")
    model = body.get("model")
    stream = bool(body.get("stream", False))

    chat_messages: list[dict[str, Any]] = []
    for text in _anthropic_system_text(body.get("system")):
        chat_messages.append({"role": "system", "content": text})
    for message in messages:
        if not isinstance(message, dict):
            raise InvalidRequestBody("anthropic messages entries must be objects")
        chat_messages.extend(_anthropic_message_to_chat(message))

    tools = _anthropic_tools_to_chat(body.get("tools"))

    base_body: dict[str, Any] = {"model": model}
    for key in ("temperature", "top_p", "max_tokens"):
        if body.get(key) is not None:
            base_body[key] = body[key]
    if body.get("stop_sequences") is not None:
        base_body["stop"] = body["stop_sequences"]
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        base_body["thinking"] = deepcopy(thinking)
    if body.get("tool_choice") is not None:
        base_body["tool_choice"] = _anthropic_tool_choice_to_chat(body["tool_choice"])
    if body.get("metadata") is not None:
        base_body["metadata"] = deepcopy(body["metadata"])

    return NormalizedRequest(
        protocol=PROTOCOL_ANTHROPIC,
        model=model,
        stream=stream,
        messages=chat_messages,
        tools=tools,
        base_body=base_body,
    )


def _responses_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") in ("input_text", "output_text", "text"):
                parts.append(b.get("text") or b.get("value") or "")
        return "\n".join(parts)
    return ""


def _responses_image_to_chat_block(block: dict[str, Any]) -> dict[str, Any] | None:
    image_url = block.get("image_url")
    if isinstance(image_url, str):
        return {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": block.get("detail") or "auto"},
        }
    if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
        return {
            "type": "image_url",
            "image_url": {"url": image_url["url"], "detail": block.get("detail") or "auto"},
        }
    return None


def _responses_message_to_chat(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    content = message.get("content")
    if role in ("developer", "system"):
        return [{"role": "system", "content": _responses_content_text(content)}]
    if isinstance(content, str):
        return [{"role": role or "user", "content": content}]
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("input_text", "output_text", "text"):
                parts.append({"type": "text", "text": block.get("text") or block.get("value") or ""})
            elif btype == "input_image":
                image = _responses_image_to_chat_block(block)
                if image is not None:
                    parts.append(image)
        return [{"role": role or "user", "content": parts}]
    return []


def _responses_item_to_chat(item: dict[str, Any]) -> list[dict[str, Any]]:
    itype = item.get("type")
    if itype == "function_call":
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": item.get("call_id") or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": item.get("name") or "",
                            "arguments": item.get("arguments") or "{}",
                        },
                    }
                ],
            }
        ]
    if itype == "function_call_output":
        return [
            {"role": "tool", "tool_call_id": item.get("call_id") or "", "content": item.get("output") or ""}
        ]
    if itype == "reasoning":
        return []
    if itype == "message":
        return _responses_message_to_chat({"role": item.get("role"), "content": item.get("content")})
    if item.get("role"):
        return _responses_message_to_chat(item)
    return []


def _responses_tools_to_chat(tools: Any, functions: Any) -> list[dict[str, Any]] | None:
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") not in (None, "function"):
            continue
        fn = tool.get("function") or {}
        name = fn.get("name") or tool.get("name")
        if not name:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": fn.get("description") or tool.get("description") or "",
                    "parameters": fn.get("parameters")
                    or tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    for fn in functions or []:
        if isinstance(fn, dict) and fn.get("name"):
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn.get("description") or "",
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
    return out


def _parse_responses(body: dict[str, Any]) -> NormalizedRequest:
    model = body.get("model")
    stream = bool(body.get("stream", False))

    chat_messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if instructions:
        chat_messages.append({"role": "system", "content": _responses_content_text(instructions)})

    input_data = body.get("input")
    if isinstance(input_data, str):
        chat_messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, dict):
                chat_messages.extend(_responses_item_to_chat(item))
    else:
        raise InvalidRequestBody("input must be a string or a list")

    tools = _responses_tools_to_chat(body.get("tools"), body.get("functions"))

    base_body: dict[str, Any] = {"model": model}
    if body.get("temperature") is not None:
        base_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        base_body["top_p"] = body["top_p"]
    if body.get("max_output_tokens") is not None:
        base_body["max_tokens"] = body["max_output_tokens"]
    elif body.get("max_tokens") is not None:
        base_body["max_tokens"] = body["max_tokens"]
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        base_body["reasoning"] = deepcopy(reasoning)
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort:
            base_body["reasoning_effort"] = effort
    elif isinstance(body.get("reasoning_effort"), str) and body["reasoning_effort"]:
        base_body["reasoning_effort"] = body["reasoning_effort"]
    if body.get("tool_choice") is not None:
        base_body["tool_choice"] = _responses_tool_choice_to_chat(body["tool_choice"])
    for key in ("parallel_tool_calls", "store", "user", "metadata", "truncation"):
        if body.get(key) is not None:
            base_body[key] = deepcopy(body[key])
    text = body.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), str):
        fmt = text["format"]
        if fmt == "json_schema":
            base_body["response_format"] = {
                "type": "json_schema",
                "json_schema": deepcopy(text.get("schema") or {}),
            }
        elif fmt == "plain_text":
            base_body["response_format"] = {"type": "text"}

    return NormalizedRequest(
        protocol=PROTOCOL_RESPONSES,
        model=model,
        stream=stream,
        messages=chat_messages,
        tools=tools,
        base_body=base_body,
    )


# ---------------------------------------------------------------------------
# Rendering: chat response -> client protocol (non-streaming)
# ---------------------------------------------------------------------------
def _stop_reason(finish: str | None) -> str:
    return {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "refusal",
    }.get(finish or "", "end_turn")


def _render_anthropic(chat_response: dict[str, Any]) -> dict[str, Any]:
    choice = (chat_response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")
    model = chat_response.get("model") or ""

    content_blocks: list[dict[str, Any]] = []
    if message.get("reasoning_content"):
        content_blocks.append({"type": "thinking", "thinking": message["reasoning_content"]})
    content = message.get("content")
    if content:
        content_blocks.append({"type": "text", "text": content})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                "name": fn.get("name") or "",
                "input": _safe_json(fn.get("arguments") or ""),
            }
        )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = chat_response.get("usage") or {}
    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": _stop_reason(finish),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _render_responses(chat_response: dict[str, Any]) -> dict[str, Any]:
    choice = (chat_response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    model = chat_response.get("model") or ""
    created = chat_response.get("created") or int(time.time())

    output: list[dict[str, Any]] = []
    if message.get("reasoning_content"):
        reasoning = message["reasoning_content"]
        output.append(
            {
                "type": "reasoning",
                "id": f"rs_{uuid.uuid4().hex[:16]}",
                "summary": [{"type": "summary_text", "text": reasoning}],
                "content": [{"type": "reasoning_text", "text": reasoning, "signature": None}],
            }
        )
    content = message.get("content")
    if content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:16]}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:16]}",
                "call_id": tc.get("id"),
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "",
                "status": "completed",
            }
        )

    usage = chat_response.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return {
        "id": f"resp_{uuid.uuid4().hex[:12]}",
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "temperature": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def render_response(protocol: str, chat_response: dict[str, Any]) -> dict[str, Any]:
    """Render a chat-format upstream response into the client's protocol."""
    if protocol == PROTOCOL_ANTHROPIC:
        return _render_anthropic(chat_response)
    if protocol == PROTOCOL_RESPONSES:
        return _render_responses(chat_response)
    return chat_response


def render_error_payload(protocol: str, error_body: dict[str, Any]) -> dict[str, Any]:
    """Render an OpenAI error envelope into the client's protocol error shape."""
    if protocol != PROTOCOL_ANTHROPIC:
        return error_body
    message = (error_body.get("error") or {}).get("message") or error_body.get("message", "")
    etype = (error_body.get("error") or {}).get("code") or "api_error"
    return {"type": "error", "error": {"type": etype, "message": message}}


# ---------------------------------------------------------------------------
# Streaming: chat SSE chunks -> client protocol SSE lines
# ---------------------------------------------------------------------------
def _anthropic_event(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _responses_event(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _accumulate_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    rid = ""
    model = ""
    created = 0
    role = "assistant"
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish: str | None = None
    for chunk in chunks:
        if chunk.get("id"):
            rid = chunk["id"]
        if chunk.get("model"):
            model = chunk["model"]
        if chunk.get("created"):
            created = chunk["created"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        delta = choice.get("delta") or {}
        if delta.get("role"):
            role = delta["role"]
        content = delta.get("content")
        if isinstance(content, str) and content:
            content_parts.append(content)
        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            index = tc.get("index", 0)
            cur = tool_calls.setdefault(index, {"id": None, "name": "", "arguments": ""})
            if tc.get("id"):
                cur["id"] = tc["id"]
            fn = tc.get("function")
            if isinstance(fn, dict):
                if fn.get("name"):
                    cur["name"] = fn["name"]
                if fn.get("arguments"):
                    cur["arguments"] += fn["arguments"]
    return {
        "id": rid,
        "model": model,
        "created": created,
        "role": role,
        "content": "".join(content_parts),
        "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
        "finish": finish,
    }


def chat_response_chunks(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a full non-stream chat response into OpenAI SSE chunk dicts."""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")
    rid = response.get("id") or ""
    model = response.get("model") or ""
    created = response.get("created") or int(time.time())
    chunks: list[dict[str, Any]] = []

    def emit(delta: dict[str, Any], finish_reason: str | None = None) -> None:
        chunks.append(
            {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    if message.get("reasoning_content"):
        emit({"reasoning_content": message["reasoning_content"]})
    content = message.get("content")
    if content:
        emit({"role": "assistant", "content": content})
    else:
        emit({"role": "assistant", "content": ""})
    tool_calls = message.get("tool_calls")
    if tool_calls:
        calls = [
            {
                "index": i,
                "id": tc.get("id"),
                "type": tc.get("type") or "function",
                "function": {
                    "name": (tc.get("function") or {}).get("name") or "",
                    "arguments": (tc.get("function") or {}).get("arguments") or "",
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
        emit({"tool_calls": calls})
    emit({}, finish_reason=finish or "stop")
    return chunks


def chat_stream_to_chat_lines(chunks: list[dict[str, Any]]) -> list[str]:
    lines = ["data: " + json.dumps(c, ensure_ascii=False) + "\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return lines


def chat_stream_to_anthropic_lines(chunks: list[dict[str, Any]]) -> list[str]:
    acc = _accumulate_chunks(chunks)
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    model = acc["model"]
    role = acc["role"]
    finish = acc["finish"]

    blocks: list[dict[str, Any]] = []
    if acc["content"]:
        blocks.append({"type": "text", "text": acc["content"]})
    for tc in acc["tool_calls"]:
        blocks.append(
            {
                "type": "tool_use",
                "id": tc["id"] or f"toolu_{uuid.uuid4().hex[:16]}",
                "name": tc["name"] or "",
                "input": _safe_json(tc["arguments"]) if tc["arguments"] else {},
                "_raw_args": tc["arguments"],
            }
        )

    lines: list[str] = []
    lines.append(
        _anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": role,
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )
    )
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            start_block: dict[str, Any] = {"type": "text", "text": ""}
            delta_payload: dict[str, Any] = {"type": "text_delta", "text": block["text"]}
        else:
            start_block = {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": {},
            }
            delta_payload = {"type": "input_json_delta", "partial_json": block["_raw_args"]}
        lines.append(
            _anthropic_event(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": start_block},
            )
        )
        lines.append(
            _anthropic_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta_payload},
            )
        )
        lines.append(_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": index}))
    lines.append(
        _anthropic_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": _stop_reason(finish), "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        )
    )
    lines.append(_anthropic_event("message_stop", {"type": "message_stop"}))
    return lines


def chat_stream_to_responses_lines(chunks: list[dict[str, Any]]) -> list[str]:
    acc = _accumulate_chunks(chunks)
    created = acc["created"] or int(time.time())
    model = acc["model"]
    finish = acc["finish"]

    lines: list[str] = []
    resp_id = f"resp_{uuid.uuid4().hex[:12]}"
    lines.append(
        _responses_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": resp_id,
                    "object": "response",
                    "created_at": created,
                    "status": "in_progress",
                    "model": model,
                    "output": [],
                },
            },
        )
    )

    output_index = 0
    if acc["content"]:
        msg_id = f"msg_{uuid.uuid4().hex[:16]}"
        text = acc["content"]
        lines.append(
            _responses_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "type": "message",
                        "id": msg_id,
                        "status": "in_progress",
                        "role": acc["role"],
                        "content": [],
                    },
                },
            )
        )
        lines.append(
            _responses_event(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": msg_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
        )
        lines.append(
            _responses_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text,
                },
            )
        )
        lines.append(
            _responses_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": msg_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                },
            )
        )
        lines.append(
            _responses_event(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": msg_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []},
                },
            )
        )
        item_done = {
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": acc["role"],
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        lines.append(
            _responses_event(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": output_index, "item": item_done},
            )
        )
        output_index += 1

    for tc in acc["tool_calls"]:
        call_id = tc["id"] or f"fc_{uuid.uuid4().hex[:12]}"
        item_id = f"fc_{uuid.uuid4().hex[:16]}"
        lines.append(
            _responses_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": tc["name"] or "",
                        "arguments": "",
                        "status": "in_progress",
                    },
                },
            )
        )
        lines.append(
            _responses_event(
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "delta": tc["arguments"],
                },
            )
        )
        lines.append(
            _responses_event(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "arguments": tc["arguments"],
                },
            )
        )
        lines.append(
            _responses_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": tc["name"] or "",
                        "arguments": tc["arguments"],
                        "status": "completed",
                    },
                },
            )
        )
        output_index += 1

    final_message: dict[str, Any] = {"role": acc["role"], "content": acc["content"] or None}
    if acc["tool_calls"]:
        final_message["tool_calls"] = [
            {
                "id": tc["id"] or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": tc["name"] or "", "arguments": tc["arguments"]},
            }
            for tc in acc["tool_calls"]
        ]
    completed = _render_responses(
        {
            "id": rid_or_empty(acc["id"]),
            "model": model,
            "created": created,
            "choices": [{"index": 0, "message": final_message, "finish_reason": finish}],
        }
    )
    completed["status"] = "completed"
    lines.append(
        _responses_event("response.completed", {"type": "response.completed", "response": completed})
    )
    return lines


def rid_or_empty(value: str) -> str:
    return value


def render_sse_lines(protocol: str, response: dict[str, Any]) -> list[str]:
    """Render a complete chat response as the client protocol's SSE stream."""
    chunks = chat_response_chunks(response)
    if protocol == PROTOCOL_ANTHROPIC:
        return chat_stream_to_anthropic_lines(chunks)
    if protocol == PROTOCOL_RESPONSES:
        return chat_stream_to_responses_lines(chunks)
    return chat_stream_to_chat_lines(chunks)


def translate_stream_lines(protocol: str, chunks: list[dict[str, Any]]) -> list[str]:
    """Translate buffered chat SSE chunks into the client protocol's SSE."""
    if protocol == PROTOCOL_ANTHROPIC:
        return chat_stream_to_anthropic_lines(chunks)
    if protocol == PROTOCOL_RESPONSES:
        return chat_stream_to_responses_lines(chunks)
    return chat_stream_to_chat_lines(chunks)
