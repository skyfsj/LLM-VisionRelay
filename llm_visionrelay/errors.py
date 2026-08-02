"""Central error hierarchy for llm-visionrelay.

All errors inherit from :class:`VisionProxyError` which carries an OpenAI-style
error code and an HTTP status code. Handlers translate these into the OpenAI
error envelope defined in the spec.
"""

from __future__ import annotations


class VisionProxyError(Exception):
    """Base error with an OpenAI-compatible code and HTTP status."""

    status_code: int = 400
    code: str = "llm_visionrelay_error"
    default_message: str = "vision proxy error"

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.status_code = status_code or self.status_code
        self.code = code or self.code
        super().__init__(self.message)


class InvalidHeader(VisionProxyError):
    code = "invalid_header"
    default_message = "invalid request header"
    status_code = 400


class MissingUpstreamBaseUrl(VisionProxyError):
    code = "upstream_base_url_missing"
    default_message = "missing X-Upstream-Base-URL request header"
    status_code = 400


class MissingAuthorization(VisionProxyError):
    code = "authorization_missing"
    default_message = "missing Authorization request header"
    status_code = 401


class MissingVisionConfig(VisionProxyError):
    code = "vision_config_missing"
    default_message = "images present but X-Vision-Base-URL / X-Vision-Model are missing"
    status_code = 400


class InvalidRequestBody(VisionProxyError):
    code = "invalid_request_body"
    default_message = "invalid request body"
    status_code = 400


class InvalidImage(VisionProxyError):
    code = "invalid_image"
    default_message = "invalid image"
    status_code = 400


class ImageTooLarge(VisionProxyError):
    code = "image_too_large"
    default_message = "image exceeds maximum allowed size"
    status_code = 413


class ImageLimitExceeded(VisionProxyError):
    code = "image_limit_exceeded"
    default_message = "too many images in request"
    status_code = 413


class TotalImageBytesExceeded(VisionProxyError):
    code = "total_image_bytes_exceeded"
    default_message = "total image bytes exceed allowed limit"
    status_code = 413


class UnsupportedMimeType(VisionProxyError):
    code = "unsupported_mime_type"
    default_message = "unsupported image mime type"
    status_code = 400


class SSRFRejected(VisionProxyError):
    code = "ssrf_rejected"
    default_message = "remote image URL rejected by SSRF protection"
    status_code = 400


class VisionTimeoutError(VisionProxyError):
    code = "vision_timeout"
    default_message = "vision model request timed out"
    status_code = 504


class VisionInvalidResponse(VisionProxyError):
    code = "vision_invalid_response"
    default_message = "vision model returned an invalid response"
    status_code = 502


class VisionAnalysisFailed(VisionProxyError):
    code = "vision_analysis_failed"
    default_message = "vision model analysis failed"
    status_code = 502


class UpstreamTimeoutError(VisionProxyError):
    code = "upstream_timeout"
    default_message = "upstream text model request timed out"
    status_code = 504


class UpstreamNonJsonError(VisionProxyError):
    code = "upstream_non_json"
    default_message = "upstream text model returned a non-JSON response"
    status_code = 502


class UpstreamRequestFailed(VisionProxyError):
    code = "upstream_request_failed"
    default_message = "upstream text model request failed"
    status_code = 502


class ToolLoopLimitExceeded(VisionProxyError):
    code = "tool_loop_limit_exceeded"
    default_message = "internal tool loop exceeded its limit"
    status_code = 400


class ToolNameConflict(VisionProxyError):
    code = "tool_name_conflict"
    default_message = "client tool conflicts with reserved __vision_ prefix"
    status_code = 400


class MixedToolCallsError(VisionProxyError):
    code = "mixed_tool_calls"
    default_message = "upstream mixed internal __vision_ tools with external tools in one round"
    status_code = 502


class CacheDatabaseError(VisionProxyError):
    code = "cache_database_error"
    default_message = "cache database error"
    status_code = 500


class InvalidImageRef(VisionProxyError):
    code = "invalid_image_ref"
    default_message = "image_ref does not belong to the current tenant/conversation"
    status_code = 400


class InvalidBBox(VisionProxyError):
    code = "invalid_bbox"
    default_message = "bbox must be [x1, y1, x2, y2] with x1 < x2 and y1 < y2 in [0, 1]"
    status_code = 400


class QueryTooLongError(VisionProxyError):
    code = "query_too_long"
    default_message = "query exceeds maximum allowed length"
    status_code = 400


class MissingManagementToken(VisionProxyError):
    code = "management_token_missing"
    default_message = "management endpoint requires a token"
    status_code = 403


class ForbiddenSource(VisionProxyError):
    code = "forbidden_source"
    default_message = "management endpoint only allows loopback sources"
    status_code = 403


def openai_error_body(err: VisionProxyError) -> dict:
    """Render an OpenAI style error envelope."""
    return {
        "error": {
            "message": err.message,
            "type": "llm_visionrelay_error",
            "param": None,
            "code": err.code,
        }
    }
