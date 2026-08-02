"""Built-in vision tool definitions and the internal tool-call loop.

The middleware executes its own ``__vision_`` tools against the local cache and
vision model, then re-asks the upstream text model with ``role=tool`` messages,
preserving ``reasoning_content`` and unknown assistant fields. Client-defined
external tools are never executed and pass straight through to the client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm_visionrelay.config import Config
from llm_visionrelay.errors import (
    InvalidBBox,
    MixedToolCallsError,
    QueryTooLongError,
    ToolNameConflict,
)
from llm_visionrelay.headers import RequestConfig
from llm_visionrelay.image_fetcher import ImageHandle, ImageService
from llm_visionrelay.models import AnalyzeToolArgs
from llm_visionrelay.upstream_protocols import UpstreamAdapter
from llm_visionrelay.vision_client import (
    VISION_TOOL_PREFIX,
    CacheCounter,
    VisionConfig,
    VisionResult,
    VisionService,
    enrich_bbox_pixels,
)

VISION_SYSTEM_HINT = (
    "你可以使用 __vision_ 前缀工具读取图片细节。\n"
    "不要在同一轮同时调用 __vision_ 工具和其他外部工具。\n"
    "已有视觉摘要足够时不要调用视觉工具。"
)

LIST_IMAGES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "__vision_list_images",
        "description": "列出当前对话中可以继续分析的图片引用和已有基础摘要。",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

ANALYZE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "__vision_analyze",
        "description": "对当前对话中已经缓存的图片执行更具体的视觉分析。仅当已有视觉摘要不足以回答问题时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_ref": {
                    "type": "string",
                    "description": "必须来自当前对话的图片引用",
                },
                "query": {
                    "type": "string",
                    "description": "需要视觉模型重点检查的具体问题",
                },
                "mode": {
                    "type": "string",
                    "enum": ["general", "ocr", "table", "diagram", "ui", "detail"],
                    "default": "detail",
                },
                "bbox": {
                    "type": "array",
                    "description": "可选的归一化裁剪区域 [x1,y1,x2,y2]，每个值范围为0到1",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "force_refresh": {"type": "boolean", "default": False},
            },
            "required": ["image_ref", "query"],
            "additionalProperties": False,
        },
    },
}

CROP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "__vision_crop",
        "description": "把当前对话中已缓存的图片裁剪到指定区域，返回裁剪后的新图片引用（image_ref），可直接用于后续 __vision_analyze 等工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_ref": {
                    "type": "string",
                    "description": "必须来自当前对话或由本中间件生成的图片引用",
                },
                "bbox": {
                    "type": "array",
                    "description": "归一化裁剪区域 [x1,y1,x2,y2]，每个值范围为0到1",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "jpeg"],
                    "default": "png",
                    "description": "输出图片格式",
                },
            },
            "required": ["image_ref", "bbox"],
            "additionalProperties": False,
        },
    },
}

RESIZE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "__vision_resize",
        "description": "把当前对话中已缓存的图片缩放到指定像素尺寸，返回缩放后的新图片引用（image_ref），可直接用于后续 __vision_analyze 等工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_ref": {
                    "type": "string",
                    "description": "必须来自当前对话或由本中间件生成的图片引用",
                },
                "width": {"type": "integer", "minimum": 1, "maximum": 8192},
                "height": {"type": "integer", "minimum": 1, "maximum": 8192},
                "format": {
                    "type": "string",
                    "enum": ["png", "jpeg"],
                    "default": "png",
                    "description": "输出图片格式",
                },
            },
            "required": ["image_ref", "width", "height"],
            "additionalProperties": False,
        },
    },
}

MASK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "__vision_mask",
        "description": "对当前对话中已缓存的图片指定区域生成蒙版（模糊/高亮/聚焦），返回处理后的新图片引用（image_ref）。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_ref": {
                    "type": "string",
                    "description": "必须来自当前对话或由本中间件生成的图片引用",
                },
                "bbox": {
                    "type": "array",
                    "description": "归一化区域 [x1,y1,x2,y2]，每个值范围为0到1",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "mode": {
                    "type": "string",
                    "enum": ["blur", "highlight", "dim"],
                    "default": "blur",
                    "description": "blur=区域模糊；highlight=区域高亮；dim=区域外压暗（聚焦）",
                },
                "format": {
                    "type": "string",
                    "enum": ["png", "jpeg"],
                    "default": "png",
                    "description": "输出图片格式",
                },
            },
            "required": ["image_ref", "bbox"],
            "additionalProperties": False,
        },
    },
}


def build_vision_tools() -> list[dict[str, Any]]:
    return [
        json.loads(json.dumps(LIST_IMAGES_TOOL)),
        json.loads(json.dumps(ANALYZE_TOOL)),
        json.loads(json.dumps(CROP_TOOL)),
        json.loads(json.dumps(RESIZE_TOOL)),
        json.loads(json.dumps(MASK_TOOL)),
    ]


def _is_vision_tool(name: str) -> bool:
    return isinstance(name, str) and name.startswith(VISION_TOOL_PREFIX)


def merge_tools(client_tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Append built-in vision tools, rejecting reserved-name collisions."""
    merged = [json.loads(json.dumps(t)) for t in (client_tools or [])]
    for tool in merged:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if _is_vision_tool(name):
            raise ToolNameConflict(f"client tool name {name!r} conflicts with reserved prefix")
    merged.extend(build_vision_tools())
    return merged


def strip_vision_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return tools
    return [t for t in tools if not _is_vision_tool((t.get("function") or {}).get("name"))]


@dataclass
class ToolLoopResult:
    response: dict[str, Any]
    status_code: int
    internal_rounds: int = 0
    vision_tool_calls: int = 0
    buffered: bool = False
    exceeded: bool = False


def _tool_error_content(code: str, message: str) -> str:
    return json.dumps({"error": {"code": code, "message": message}}, ensure_ascii=False)


class ToolLoop:
    def __init__(
        self,
        config: Config,
        upstream: UpstreamAdapter,
        vision_service: VisionService,
        image_service: ImageService,
    ) -> None:
        self.config = config
        self.upstream = upstream
        self.vision_service = vision_service
        self.image_service = image_service

    async def run(
        self,
        cfg: RequestConfig,
        vision_cfg: VisionConfig | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        image_handles: list[ImageHandle],
        counter: CacheCounter,
        base_body: dict[str, Any],
        ttl: float | None = None,
        initial_response: dict[str, Any] | None = None,
    ) -> ToolLoopResult:
        rounds = 0
        vision_tool_calls = 0
        buffered = initial_response is not None
        response = initial_response
        self._last_status = 200
        self._derived_images: dict[str, ImageHandle] = {}

        while True:
            if response is None:
                resp = await self.upstream.request_json(cfg, self._payload(base_body, messages, tools))
                self._last_status = resp.status_code
                response = resp.body

            if isinstance(response, dict) and response.get("error") is not None:
                return ToolLoopResult(response, self._last_status, rounds, vision_tool_calls, buffered)

            message = self._extract_message(response)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return ToolLoopResult(response, self._status(), rounds, vision_tool_calls, buffered)

            vision_tc = [t for t in tool_calls if _is_vision_tool((t.get("function") or {}).get("name"))]
            external_tc = [
                t for t in tool_calls if not _is_vision_tool((t.get("function") or {}).get("name"))
            ]
            if vision_tc and external_tc:
                raise MixedToolCallsError()
            if external_tc:
                return ToolLoopResult(response, self._status(), rounds, vision_tool_calls, buffered)

            rounds += 1
            vision_tool_calls += len(vision_tc)
            if (
                rounds > self.config.max_tool_rounds
                or vision_tool_calls > self.config.max_tool_calls_per_request
            ):
                messages.append(_deepcopy(message))
                for tc in vision_tc:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": _tool_error_content(
                                "tool_loop_limit_exceeded",
                                "达到内部工具调用限制，禁止继续调用内置视觉工具",
                            ),
                        }
                    )
                final_tools = strip_vision_tools(tools)
                resp = await self.upstream.request_json(cfg, self._payload(base_body, messages, final_tools))
                self._last_status = resp.status_code
                response = resp.body
                return ToolLoopResult(
                    response, self._status(), rounds, vision_tool_calls, buffered, exceeded=True
                )

            messages.append(_deepcopy(message))
            for tc in vision_tc:
                result_content = await self._execute_tool(cfg, vision_cfg, tc, image_handles, counter, ttl)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_content})
            response = None

    # ------------------------------------------------------------------ upstream
    def _payload(
        self, base_body: dict[str, Any], messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        payload = _deepcopy(base_body)
        payload["messages"] = _deepcopy(messages)
        if tools is not None:
            payload["tools"] = _deepcopy(tools)
        else:
            payload.pop("tools", None)
        payload.pop("stream", None)
        payload["stream"] = False
        return payload

    def _extract_message(self, response: dict[str, Any]) -> dict[str, Any]:
        try:
            return response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise MixedToolCallsError("upstream response missing choices[0].message") from exc

    def _status(self) -> int:
        return self._last_status

    # ------------------------------------------------------------------ tools
    def _image_ref_available(self, image_ref: str, image_handles: list[ImageHandle]) -> bool:
        if any(h.image_ref == image_ref for h in image_handles):
            return True
        if image_ref in self._derived_images:
            return True
        return False

    async def _execute_tool(
        self,
        cfg: RequestConfig,
        vision_cfg: VisionConfig | None,
        tool_call: dict[str, Any],
        image_handles: list[ImageHandle],
        counter: CacheCounter,
        ttl: float | None,
    ) -> str:
        fn = tool_call.get("function") or {}
        name = fn.get("name") or ""
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            return _tool_error_content("invalid_arguments", "工具参数不是合法 JSON")

        if name == "__vision_list_images":
            return json.dumps(
                {"images": await self._list_images(cfg, vision_cfg, image_handles)}, ensure_ascii=False
            )
        if name == "__vision_analyze":
            return await self._analyze(cfg, vision_cfg, args, image_handles, counter, ttl)
        if name in ("__vision_crop", "__vision_resize", "__vision_mask"):
            return await self._image_process(cfg, name, args, image_handles)
        return _tool_error_content("unknown_tool", f"未知内置工具 {name!r}")

    async def _list_images(
        self, cfg: RequestConfig, vision_cfg: VisionConfig | None, image_handles: list[ImageHandle]
    ) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        for handle in image_handles:
            summary_text = ""
            if vision_cfg is not None:
                cached = await self.vision_service.get_summary_cached_only(cfg.tenant_id, handle, vision_cfg)
                if cached is not None:
                    summary_text = cached.json.get("summary") or cached.text
            images.append(
                {
                    "image_ref": handle.image_ref,
                    "summary": summary_text,
                    "mime_type": handle.mime_type,
                    "width": handle.width,
                    "height": handle.height,
                }
            )
        return images

    async def _analyze(
        self,
        cfg: RequestConfig,
        vision_cfg: VisionConfig | None,
        args: dict[str, Any],
        image_handles: list[ImageHandle],
        counter: CacheCounter,
        ttl: float | None,
    ) -> str:
        if vision_cfg is None:
            return _tool_error_content("vision_config_missing", "缺少视觉模型配置")
        try:
            parsed = AnalyzeToolArgs.model_validate(args)
        except Exception as exc:
            return _tool_error_content("invalid_arguments", f"工具参数校验失败: {exc}")

        if not self._image_ref_available(parsed.image_ref, image_handles):
            return _tool_error_content(
                "invalid_image_ref",
                "image_ref 不属于当前对话，不允许读取",
            )
        if len(parsed.query) > self.config.query_max_length:
            raise QueryTooLongError()
        if parsed.mode not in {"general", "ocr", "table", "diagram", "ui", "detail"}:
            return _tool_error_content("invalid_arguments", f"未知 mode {parsed.mode!r}")

        data, real_handle = await self.image_service.read_for_analyze(cfg.tenant_id, parsed.image_ref)

        if parsed.bbox is not None:
            bbox = parsed.bbox
            if (
                len(bbox) != 4
                or bbox[0] >= bbox[2]
                or bbox[1] >= bbox[3]
                or any(v < 0 or v > 1 for v in bbox)
            ):
                raise InvalidBBox()

        result: VisionResult = await self.vision_service.analyze(
            cfg.tenant_id,
            real_handle,
            vision_cfg,
            parsed.query,
            parsed.mode,
            parsed.bbox,
            counter,
            force_refresh=parsed.force_refresh,
            ttl=ttl,
        )
        enriched = enrich_bbox_pixels(result.json, real_handle.width, real_handle.height)
        return json.dumps(
            {
                "image_ref": parsed.image_ref,
                "query": parsed.query,
                "mode": parsed.mode,
                "cache": "hit" if result.cache_hit else "miss",
                "image_width": real_handle.width,
                "image_height": real_handle.height,
                "result": enriched,
            },
            ensure_ascii=False,
        )

    async def _image_process(
        self,
        cfg: RequestConfig,
        name: str,
        args: dict[str, Any],
        image_handles: list[ImageHandle],
    ) -> str:
        from llm_visionrelay import imaging_tools

        image_ref = args.get("image_ref")
        if not isinstance(image_ref, str) or not self._image_ref_available(image_ref, image_handles):
            return _tool_error_content("invalid_image_ref", "image_ref 不属于当前对话，不允许读取")
        fmt = args.get("format") or "png"
        if fmt not in ("png", "jpeg"):
            return _tool_error_content("invalid_arguments", f"无效输出格式 {fmt!r}")

        try:
            data, handle = await self.image_service.read_for_analyze(cfg.tenant_id, image_ref)
        except Exception:
            return _tool_error_content("invalid_image_ref", "图片读取失败或不属于当前租户")

        try:
            if name == "__vision_crop":
                if not isinstance(args.get("bbox"), list):
                    return _tool_error_content("invalid_arguments", "缺少 bbox")
                processed = imaging_tools.crop_image(data, args["bbox"], fmt)
            elif name == "__vision_resize":
                width = args.get("width")
                height = args.get("height")
                if not isinstance(width, int) or not isinstance(height, int):
                    return _tool_error_content("invalid_arguments", "width/height 必须为整数")
                processed = imaging_tools.resize_image(data, width, height, fmt)
            elif name == "__vision_mask":
                if not isinstance(args.get("bbox"), list):
                    return _tool_error_content("invalid_arguments", "缺少 bbox")
                mode = args.get("mode") or "blur"
                processed = imaging_tools.mask_image(data, args["bbox"], mode, fmt)
            else:
                return _tool_error_content("unknown_tool", f"未知内置工具 {name!r}")
        except Exception as exc:
            return _tool_error_content("image_processing_failed", f"图片处理失败: {exc}")

        new_handle = await self.image_service.register_processed(
            cfg.tenant_id,
            processed,
            f"image/{'jpeg' if fmt == 'jpeg' else 'png'}",
            f"derived:{name}",
            handle.detail,
        )
        self._derived_images[new_handle.image_ref] = new_handle
        return json.dumps(
            {
                "image_ref": new_handle.image_ref,
                "source_image_ref": image_ref,
                "mime_type": new_handle.mime_type,
                "width": new_handle.width,
                "height": new_handle.height,
                "size_bytes": new_handle.size_bytes,
            },
            ensure_ascii=False,
        )


def _deepcopy(value: Any) -> Any:
    return json.loads(json.dumps(value))
