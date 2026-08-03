"""Header parsing unit tests."""

from __future__ import annotations

import pytest
from llm_visionrelay.config import Config
from llm_visionrelay.errors import (
    InvalidHeader,
    MissingAuthorization,
    MissingUpstreamBaseUrl,
)
from llm_visionrelay.headers import parse_bool, parse_request_headers


def _cfg() -> Config:
    return Config()


def _headers(**kw) -> dict:
    base = {
        "Authorization": "Bearer TEXT_KEY",
        "X-Upstream-Base-URL": "https://api.deepseek.com",
    }
    base.update(kw)
    return base


def test_parse_bool_variants() -> None:
    assert parse_bool("true", "x", False) is True
    assert parse_bool("TRUE", "x", False) is True
    assert parse_bool("1", "x", False) is True
    assert parse_bool("yes", "x", False) is True
    assert parse_bool("false", "x", True) is False
    assert parse_bool("0", "x", True) is False
    assert parse_bool("no", "x", True) is False
    assert parse_bool(None, "x", True) is True
    with pytest.raises(InvalidHeader):
        parse_bool("maybe", "x", True)


def test_defaults() -> None:
    cfg = parse_request_headers(_headers(), _cfg())
    assert cfg.upstream_base_url == "https://api.deepseek.com"
    assert cfg.authorization == "Bearer TEXT_KEY"
    assert cfg.auto_analyze is True
    assert cfg.tools_enabled is True
    assert cfg.cache_ttl == 30 * 24 * 3600
    assert cfg.force_refresh is False
    assert cfg.vision_ready is False


def test_missing_upstream() -> None:
    with pytest.raises(MissingUpstreamBaseUrl):
        parse_request_headers({"Authorization": "Bearer X"}, _cfg())


def test_missing_authorization() -> None:
    with pytest.raises(MissingAuthorization):
        parse_request_headers({"X-Upstream-Base-URL": "https://x.example.com"}, _cfg())


def test_base_url_trailing_slash_normalized() -> None:
    cfg = parse_request_headers(_headers(**{"X-Upstream-Base-URL": "https://api.deepseek.com/"}), _cfg())
    assert cfg.upstream_base_url == "https://api.deepseek.com"


def test_invalid_base_url() -> None:
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**{"X-Upstream-Base-URL": "not-a-url"}), _cfg())
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**{"X-Upstream-Base-URL": "ftp://x.example.com"}), _cfg())


def test_upstream_model_override() -> None:
    cfg = parse_request_headers(_headers(**{"X-Upstream-Model": "deepseek-reasoner"}), _cfg())
    assert cfg.upstream_model == "deepseek-reasoner"


def test_tenant_derived_from_authorization_digest() -> None:
    from llm_visionrelay.security import tenant_id_from_authorization

    cfg = parse_request_headers(_headers(), _cfg())
    assert cfg.tenant_id == tenant_id_from_authorization("Bearer TEXT_KEY")
    assert cfg.tenant_id != "Bearer TEXT_KEY"


def test_tenant_derived_from_namespace() -> None:
    from llm_visionrelay.security import tenant_id_from_namespace

    cfg = parse_request_headers(_headers(**{"X-Vision-Cache-Namespace": "client-42"}), _cfg())
    assert cfg.tenant_id == tenant_id_from_namespace("client-42")


def test_vision_config_parsed() -> None:
    cfg = parse_request_headers(
        _headers(
            **{
                "X-Vision-Base-URL": "https://vision.example.com/v1/",
                "X-Vision-Model": "qwen-vl",
                "X-Vision-Authorization": "Bearer VISION_KEY",
            }
        ),
        _cfg(),
    )
    assert cfg.vision_base_url == "https://vision.example.com/v1"
    assert cfg.vision_model == "qwen-vl"
    assert cfg.vision_authorization == "Bearer VISION_KEY"
    assert cfg.vision_ready is True


def test_custom_vision_headers() -> None:
    cfg = parse_request_headers(
        _headers(
            **{
                "X-Vision-Header-X-API-Key": "abc",
                "X-Vision-Header-Custom-Token": "xyz",
            }
        ),
        _cfg(),
    )
    assert cfg.vision_headers == {"x-api-key": "abc", "custom-token": "xyz"}


@pytest.mark.parametrize(
    "extra",
    [
        {"X-Vision-Header-Host": "evil.example.com"},
        {"X-Vision-Header-Content-Length": "5"},
        {"X-Vision-Header-Connection": "close"},
        {"X-Vision-Header-Transfer-Encoding": "chunked"},
        {"X-Vision-Header-Content-Type": "text/html"},
        {"X-Vision-Header-Authorization": "Bearer X"},
    ],
)
def test_forbidden_vision_header_override(extra: dict) -> None:
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**extra), _cfg())


def test_vision_header_count_limit() -> None:
    cfg = Config(max_vision_headers=2)
    headers = _headers(**{f"X-Vision-Header-X-H{i}": str(i) for i in range(3)})
    with pytest.raises(InvalidHeader):
        parse_request_headers(headers, cfg)


def test_vision_header_length_limits() -> None:
    cfg = Config(max_vision_header_value_length=8)
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**{"X-Vision-Header-X-Long": "a" * 9}), cfg)
    cfg2 = Config(max_vision_header_name_length=8)
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**{"X-Vision-Header-ThisNameIsWayTooLong": "v"}), cfg2)


def test_cache_ttl_parsing() -> None:
    cfg = parse_request_headers(_headers(**{"X-Vision-Cache-TTL": "86400"}), _cfg())
    assert cfg.cache_ttl == 86400.0
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**{"X-Vision-Cache-TTL": "abc"}), _cfg())


def test_feature_flags() -> None:
    cfg = parse_request_headers(
        _headers(
            **{
                "X-Vision-Auto-Analyze": "false",
                "X-Vision-Tools": "0",
                "X-Vision-Force-Refresh": "yes",
            }
        ),
        _cfg(),
    )
    assert cfg.auto_analyze is False
    assert cfg.tools_enabled is False
    assert cfg.force_refresh is True


def test_vision_params_parsed() -> None:
    from llm_visionrelay.security import sha256_hex

    cfg = parse_request_headers(
        _headers(**{"X-Vision-Params": '{"reasoning_effort": "high", "temperature": 0.3}'}), _cfg()
    )
    assert cfg.vision_params == {"reasoning_effort": "high", "temperature": 0.3}
    assert cfg.vision_params_hash == sha256_hex('{"reasoning_effort":"high","temperature":0.3}')


def test_vision_params_default_empty() -> None:
    cfg = parse_request_headers(_headers(), _cfg())
    assert cfg.vision_params == {}
    assert cfg.vision_params_hash == ""


def test_vision_params_canonical_hash_order_independent() -> None:
    from llm_visionrelay.security import sha256_hex

    a = parse_request_headers(_headers(**{"X-Vision-Params": '{"a":1,"b":2}'}), _cfg())
    b = parse_request_headers(_headers(**{"X-Vision-Params": '{"b":2,"a":1}'}), _cfg())
    assert a.vision_params_hash == b.vision_params_hash == sha256_hex('{"a":1,"b":2}')


@pytest.mark.parametrize(
    "value",
    [
        '{"model": "x"}',
        '{"messages": []}',
        "not-json",
        "[1, 2]",
        '"string"',
    ],
)
def test_vision_params_invalid(value: str) -> None:
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**{"X-Vision-Params": value}), _cfg())


def test_vision_params_too_long() -> None:
    value = '{"padding": "' + "x" * 3000 + '"}'
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**{"X-Vision-Params": value}), _cfg())


def test_upstream_protocol_parsed_and_validated() -> None:
    cfg = parse_request_headers(_headers(**{"X-Upstream-Protocol": "anthropic"}), _cfg())
    assert cfg.upstream_protocol == "anthropic"
    cfg2 = parse_request_headers(_headers(), _cfg())
    assert cfg2.upstream_protocol == "chat"
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**{"X-Upstream-Protocol": "ftp"}), _cfg())


def test_vision_max_override_headers() -> None:
    cfg = parse_request_headers(
        _headers(
            **{
                "X-Vision-Max-Images": "100",
                "X-Vision-Max-Image-Bytes": "5",
                "X-Vision-Max-Total-Image-Bytes": "200",
            }
        ),
        _cfg(),
    )
    assert cfg.max_images == 100
    assert cfg.max_image_bytes == 5 * 1024 * 1024
    assert cfg.max_total_image_bytes == 200 * 1024 * 1024


def test_vision_max_override_absent() -> None:
    cfg = parse_request_headers(_headers(), _cfg())
    assert cfg.max_images is None
    assert cfg.max_image_bytes is None
    assert cfg.max_total_image_bytes is None


@pytest.mark.parametrize("header,value", [("X-Vision-Max-Images", "0"), ("X-Vision-Max-Images", "99999")])
def test_vision_max_override_out_of_range(header: str, value: str) -> None:
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**{header: value}), _cfg())


def test_vision_max_override_non_int() -> None:
    with pytest.raises(InvalidHeader):
        parse_request_headers(_headers(**{"X-Vision-Max-Images": "many"}), _cfg())


def test_passthrough_headers_parsed() -> None:
    cfg = parse_request_headers(
        {
            "Authorization": "Bearer TEXT_KEY",
            "X-Upstream-Base-URL": "https://api.deepseek.com",
            "User-Agent": "my-agent/1.0",
            "X-Custom-Trace": "abc",
            "X-Vision-Model": "qwen",
            "Cookie": "session=secret",
            "X-Forwarded-For": "10.0.0.1",
            "Host": "example.com",
        },
        _cfg(),
    )
    pt = cfg.passthrough_headers
    assert pt.get("user-agent") == "my-agent/1.0"
    assert pt.get("x-custom-trace") == "abc"
    # full passthrough: cookie and other client headers are forwarded
    assert pt.get("cookie") == "session=secret"
    # middleware settings / IP / protocol-managed are excluded
    assert "x-vision-model" not in pt
    assert "x-forwarded-for" not in pt
    assert "host" not in pt
    # authorization is carried by cfg.authorization, never duplicated in passthrough
    assert "authorization" not in pt
    assert cfg.authorization == "Bearer TEXT_KEY"
