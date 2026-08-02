"""Protocol adapter unit tests: parsing, rendering, streaming translation."""

from __future__ import annotations

import json

from llm_visionrelay.config import Config
from llm_visionrelay.protocols import (
    PROTOCOL_ANTHROPIC,
    PROTOCOL_CHAT,
    PROTOCOL_RESPONSES,
    chat_response_chunks,
    chat_stream_to_anthropic_lines,
    chat_stream_to_responses_lines,
    parse_request,
    protocol_from_path,
    render_error_payload,
    render_response,
    render_sse_lines,
    translate_stream_lines,
)

CONFIG = Config()


def _chat_chunks(
    text: str | None = "Hello", tool_calls: list | None = None, finish: str = "stop"
) -> list[dict]:
    base = {"id": "chatcmpl-9", "object": "chat.completion.chunk", "created": 123, "model": "deepseek-chat"}
    chunks = []
    if text is not None:
        chunks.append(
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}
                ],
            }
        )
    for tc in tool_calls or []:
        chunks.append(
            {**base, "choices": [{"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": None}]}
        )
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    return chunks


def _chat_response(content: str = "hi", tool_calls: list | None = None, finish: str = "stop") -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 123,
        "model": "deepseek-chat",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }


# ------------------------------------------------------------------ detection
def test_protocol_from_path() -> None:
    assert protocol_from_path("/v1/chat/completions") == PROTOCOL_CHAT
    assert protocol_from_path("/v1/messages") == PROTOCOL_ANTHROPIC
    assert protocol_from_path("/v1/responses") == PROTOCOL_RESPONSES
    assert protocol_from_path("/v1/messages/") == PROTOCOL_ANTHROPIC
    assert protocol_from_path("/healthz") == PROTOCOL_CHAT


# ------------------------------------------------------------------ parsing: anthropic
def test_parse_anthropic_basic() -> None:
    body = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 1024,
        "system": "你是助手",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看这张"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                    },
                ],
            }
        ],
    }
    req = parse_request(PROTOCOL_ANTHROPIC, body, CONFIG)
    assert req.protocol == PROTOCOL_ANTHROPIC
    assert req.model == "claude-3-5-sonnet"
    assert req.stream is False
    assert req.messages[0] == {"role": "system", "content": "你是助手"}
    user = req.messages[1]
    assert user["role"] == "user"
    assert user["content"][0] == {"type": "text", "text": "看这张"}
    assert user["content"][1]["type"] == "image_url"
    assert user["content"][1]["image_url"]["url"] == "data:image/png;base64,AAAA"
    assert req.base_body["max_tokens"] == 1024


def test_parse_anthropic_tool_use_and_result() -> None:
    body = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "我来查"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "北京"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "晴天"}],
            },
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "查天气",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    }
    req = parse_request(PROTOCOL_ANTHROPIC, body, CONFIG)
    assert req.messages[0]["role"] == "assistant"
    assert req.messages[0]["content"] == "我来查"
    assert req.messages[0]["tool_calls"][0]["id"] == "toolu_1"
    assert req.messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(req.messages[0]["tool_calls"][0]["function"]["arguments"]) == {"city": "北京"}
    assert req.messages[1] == {"role": "tool", "tool_call_id": "toolu_1", "content": "晴天"}
    assert req.tools[0]["function"]["name"] == "get_weather"
    assert req.tools[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_parse_anthropic_image_url_and_stop() -> None:
    body = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "url", "url": "http://img.test/a.png"}}],
            }
        ],
        "stop_sequences": ["STOP"],
    }
    req = parse_request(PROTOCOL_ANTHROPIC, body, CONFIG)
    assert req.messages[0]["content"][0]["image_url"]["url"] == "http://img.test/a.png"
    assert req.base_body["stop"] == ["STOP"]


def test_parse_anthropic_invalid_no_messages() -> None:
    import pytest
    from llm_visionrelay.errors import InvalidRequestBody

    with pytest.raises(InvalidRequestBody):
        parse_request(PROTOCOL_ANTHROPIC, {"model": "m"}, CONFIG)


# ------------------------------------------------------------------ parsing: responses
def test_parse_responses_string_input() -> None:
    body = {"model": "gpt-4o", "instructions": "你是助手", "input": "你好"}
    req = parse_request(PROTOCOL_RESPONSES, body, CONFIG)
    assert req.messages[0] == {"role": "system", "content": "你是助手"}
    assert req.messages[1] == {"role": "user", "content": "你好"}


def test_parse_responses_items() -> None:
    body = {
        "model": "gpt-4o",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "看"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            },
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "晴天"},
        ],
        "tools": [
            {"type": "function", "name": "get_weather", "description": "d", "parameters": {"type": "object"}}
        ],
    }
    req = parse_request(PROTOCOL_RESPONSES, body, CONFIG)
    user = req.messages[0]
    assert user["role"] == "user"
    assert user["content"][0] == {"type": "text", "text": "看"}
    assert user["content"][1]["type"] == "image_url"
    assert req.messages[1]["role"] == "assistant"
    assert req.messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert req.messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "晴天"}
    assert req.tools[0]["function"]["name"] == "get_weather"


def test_parse_responses_invalid_input() -> None:
    import pytest
    from llm_visionrelay.errors import InvalidRequestBody

    with pytest.raises(InvalidRequestBody):
        parse_request(PROTOCOL_RESPONSES, {"model": "m", "input": 42}, CONFIG)


# ------------------------------------------------------------------ rendering
def test_render_anthropic_text_and_tool() -> None:
    chat = _chat_response(
        content="hi",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"北京"}'},
            }
        ],
        finish="tool_calls",
    )
    out = render_response(PROTOCOL_ANTHROPIC, chat)
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0] == {"type": "text", "text": "hi"}
    assert out["content"][1]["type"] == "tool_use"
    assert out["content"][1]["name"] == "get_weather"
    assert out["content"][1]["input"] == {"city": "北京"}


def test_render_anthropic_stop_reason() -> None:
    out = render_response(PROTOCOL_ANTHROPIC, _chat_response(finish="stop"))
    assert out["stop_reason"] == "end_turn"
    out = render_response(PROTOCOL_ANTHROPIC, _chat_response(finish="length"))
    assert out["stop_reason"] == "max_tokens"


def test_render_responses() -> None:
    chat = _chat_response(
        content="hi",
        tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        ],
        finish="tool_calls",
    )
    out = render_response(PROTOCOL_RESPONSES, chat)
    assert out["object"] == "response"
    assert out["status"] == "completed"
    assert out["output"][0]["type"] == "message"
    assert out["output"][0]["content"][0]["type"] == "output_text"
    assert out["output"][0]["content"][0]["text"] == "hi"
    assert out["output"][1]["type"] == "function_call"
    assert out["output"][1]["call_id"] == "call_1"
    assert out["output"][1]["name"] == "get_weather"


def test_render_responses_reasoning() -> None:
    chat = {
        "id": "c",
        "model": "m",
        "created": 1,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "答案", "reasoning_content": "思考"},
                "finish_reason": "stop",
            }
        ],
    }
    out = render_response(PROTOCOL_RESPONSES, chat)
    assert out["output"][0]["type"] == "reasoning"
    assert out["output"][1]["type"] == "message"


def test_render_error_payload() -> None:
    err = render_error_payload(PROTOCOL_ANTHROPIC, {"error": {"message": "boom", "code": "vision_timeout"}})
    assert err == {"type": "error", "error": {"type": "vision_timeout", "message": "boom"}}
    # chat and responses keep the OpenAI envelope
    assert render_error_payload(PROTOCOL_CHAT, {"error": {"message": "boom"}}) == {
        "error": {"message": "boom"}
    }


# ------------------------------------------------------------------ streaming
def test_anthropic_stream_translation() -> None:
    chunks = _chat_chunks("Hello world")
    lines = chat_stream_to_anthropic_lines(chunks)
    joined = "\n".join(lines)
    assert lines[0].startswith("event: message_start")
    assert "event: content_block_start" in joined
    assert '"type": "text_delta"' in joined
    assert '"text": "Hello world"' in joined
    assert lines[-2].startswith("event: message_delta")
    assert '"stop_reason": "end_turn"' in lines[-2]
    assert lines[-1].startswith("event: message_stop")


def test_anthropic_stream_tool_use() -> None:
    tc = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "__vision_analyze", "arguments": '{"q":"x"}'},
    }
    lines = chat_stream_to_anthropic_lines(_chat_chunks(None, [tc], finish="tool_calls"))
    joined = "\n".join(lines)
    assert '"type": "tool_use"' in joined
    assert '"type": "input_json_delta"' in joined
    assert lines[-2].index("stop_reason") > -1 and "tool_use" in lines[-2]


def test_responses_stream_translation() -> None:
    chunks = _chat_chunks("Hello world")
    lines = chat_stream_to_responses_lines(chunks)
    joined = "\n".join(lines)
    assert lines[0].startswith("event: response.created")
    assert "event: response.output_text.delta" in joined
    assert '"delta": "Hello world"' in joined
    assert lines[-1].startswith("event: response.completed")


def test_responses_stream_tool_use() -> None:
    tc = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }
    lines = chat_stream_to_responses_lines(_chat_chunks(None, [tc], finish="tool_calls"))
    joined = "\n".join(lines)
    assert "event: response.function_call_arguments.delta" in joined
    assert '"type": "function_call"' in joined


def test_translate_stream_lines_matches_protocol() -> None:
    chunks = _chat_chunks("hi")
    assert translate_stream_lines(PROTOCOL_ANTHROPIC, chunks)[-1].startswith("event: message_stop")
    assert translate_stream_lines(PROTOCOL_RESPONSES, chunks)[-1].startswith("event: response.completed")
    chat_lines = translate_stream_lines(PROTOCOL_CHAT, chunks)
    assert chat_lines[-1] == "data: [DONE]\n\n"


def test_render_sse_lines_from_response() -> None:
    response = _chat_response(content="final")
    anthropic_lines = render_sse_lines(PROTOCOL_ANTHROPIC, response)
    assert anthropic_lines[-1].startswith("event: message_stop")
    assert '"text": "final"' in "\n".join(anthropic_lines)
    responses_lines = render_sse_lines(PROTOCOL_RESPONSES, response)
    assert responses_lines[-1].startswith("event: response.completed")
    assert '"delta": "final"' in "\n".join(responses_lines)


def test_chat_response_chunks_roundtrip() -> None:
    response = _chat_response(
        content="final",
        tool_calls=[{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        finish="tool_calls",
    )
    chunks = chat_response_chunks(response)
    assert chunks[0]["choices"][0]["delta"]["content"] == "final"
    assert chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "f"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ------------------------------------------------------------------ reasoning effort passthrough
def test_parse_responses_reasoning_effort_preserved() -> None:
    body = {
        "model": "gpt-4o",
        "input": "hi",
        "reasoning": {"effort": "high", "summary": "auto"},
    }
    req = parse_request(PROTOCOL_RESPONSES, body, CONFIG)
    assert req.base_body["reasoning_effort"] == "high"
    assert req.base_body["reasoning"] == {"effort": "high", "summary": "auto"}


def test_parse_responses_top_level_reasoning_effort() -> None:
    req = parse_request(PROTOCOL_RESPONSES, {"model": "m", "input": "hi", "reasoning_effort": "low"}, CONFIG)
    assert req.base_body["reasoning_effort"] == "low"


def test_parse_chat_reasoning_effort_preserved() -> None:
    req = parse_request(PROTOCOL_CHAT, {"model": "m", "messages": [], "reasoning_effort": "medium"}, CONFIG)
    assert req.base_body["reasoning_effort"] == "medium"


def test_parse_anthropic_thinking_preserved() -> None:
    body = {
        "model": "claude",
        "max_tokens": 100,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "messages": [{"role": "user", "content": "hi"}],
    }
    req = parse_request(PROTOCOL_ANTHROPIC, body, CONFIG)
    assert req.base_body["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_render_responses_reasoning_from_effort() -> None:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "xhigh"}
    from llm_visionrelay.upstream_protocols import render_chat_to_responses

    body = render_chat_to_responses(payload)
    assert body["reasoning"] == {"effort": "xhigh"}


def test_render_anthropic_thinking_from_effort() -> None:
    from llm_visionrelay.upstream_protocols import render_chat_to_anthropic

    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"}
    body = render_chat_to_anthropic(payload)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    # preserving an explicit thinking config wins over effort mapping
    payload2 = {
        "model": "m",
        "messages": [],
        "reasoning_effort": "high",
        "thinking": {"type": "enabled", "budget_tokens": 999},
    }
    body2 = render_chat_to_anthropic(payload2)
    assert body2["thinking"] == {"type": "enabled", "budget_tokens": 999}


# ------------------------------------------------------------------ broader passthrough audit
def test_render_anthropic_max_completion_tokens() -> None:
    from llm_visionrelay.upstream_protocols import render_chat_to_anthropic

    body = render_chat_to_anthropic({"model": "m", "messages": [], "max_completion_tokens": 5000})
    assert body["max_tokens"] == 5000
    body2 = render_chat_to_anthropic({"model": "m", "messages": [], "max_tokens": 3000})
    assert body2["max_tokens"] == 3000


def test_render_responses_max_completion_tokens() -> None:
    from llm_visionrelay.upstream_protocols import render_chat_to_responses

    body = render_chat_to_responses({"model": "m", "messages": [], "max_completion_tokens": 5000})
    assert body["max_output_tokens"] == 5000


def test_tool_choice_mapping_anthropic() -> None:
    from llm_visionrelay.upstream_protocols import render_chat_to_anthropic

    body = render_chat_to_anthropic(
        {"model": "m", "messages": [], "tool_choice": {"type": "function", "function": {"name": "f"}}}
    )
    assert body["tool_choice"] == {"type": "tool", "name": "f"}
    body2 = render_chat_to_anthropic({"model": "m", "messages": [], "tool_choice": "required"})
    assert body2["tool_choice"] == {"type": "any"}


def test_anthropic_tool_choice_parsed_to_chat() -> None:
    body = {
        "model": "claude",
        "max_tokens": 100,
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    req = parse_request(PROTOCOL_ANTHROPIC, body, CONFIG)
    assert req.base_body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_responses_extra_fields_preserved() -> None:
    body = {
        "model": "gpt-4o",
        "input": "hi",
        "store": True,
        "user": "u1",
        "metadata": {"k": "v"},
        "parallel_tool_calls": False,
        "text": {"format": "json_schema", "schema": {"type": "object"}},
    }
    req = parse_request(PROTOCOL_RESPONSES, body, CONFIG)
    assert req.base_body["store"] is True
    assert req.base_body["user"] == "u1"
    assert req.base_body["metadata"] == {"k": "v"}
    assert req.base_body["parallel_tool_calls"] is False
    assert req.base_body["response_format"] == {"type": "json_schema", "json_schema": {"type": "object"}}


def test_render_responses_response_format() -> None:
    from llm_visionrelay.upstream_protocols import render_chat_to_responses

    body = render_chat_to_responses(
        {"model": "m", "messages": [], "response_format": {"type": "json_object"}}
    )
    assert body["text"] == {"format": "json_schema"}


def test_render_anthropic_thinking_block_from_reasoning() -> None:
    from llm_visionrelay.protocols import render_response

    chat = {
        "id": "c",
        "model": "m",
        "created": 1,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "答案", "reasoning_content": "思考"},
                "finish_reason": "stop",
            }
        ],
    }
    out = render_response(PROTOCOL_ANTHROPIC, chat)
    assert out["content"][0]["type"] == "thinking"
    assert out["content"][0]["thinking"] == "思考"
    assert out["content"][1]["type"] == "text"
