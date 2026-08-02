# LLM-VisionRelay

> [English](README.md) | 中文

一个 Python 异步 HTTP 中间件，让任意**纯文本**聊天模型具备视觉能力。

`LLM-VisionRelay` 接收来自 OpenAI Chat Completions、Anthropic Messages 或
OpenAI Responses 客户端的图片请求，交给**视觉模型**生成结构化文字描述，把图片块
替换为该描述，再转发给纯文本模型——并始终以客户端原本的协议返回。

## 特性

- **多协议输入**，按路径自动识别：
  - `POST /v1/chat/completions`（OpenAI Chat Completions）
  - `POST /v1/messages`（Anthropic Messages）
  - `POST /v1/responses`（OpenAI Responses）
- **视觉提取流水线** —— 拉取图片（base64 data URL、`http(s)://`），存入
  **内容寻址缓存**，交给视觉模型生成摘要，再把图片块替换为「不可信」文本上下文
  后发给文本模型。
- **多层缓存** —— SQLite 元数据 + 内容寻址文件存储 + 进程内 LRU +
  `singleflight` 并发去重。同一张图只调用一次视觉模型；同一 URL 只在别名过期后重新下载。
- **租户隔离** —— 缓存键使用 `Authorization`（或 `X-Vision-Cache-Namespace`）
  的 SHA-256 摘要，原始 Key 永不落盘；租户之间无法读取彼此的图片或结果。
- **内置视觉工具** —— 中间件注入 `__vision_` 工具并自行执行：
  - `__vision_list_images`、`__vision_analyze`
  - `__vision_crop`、`__vision_resize`、`__vision_mask`（基于 Pillow 的真实图像处理）
- **上游协议支持** —— 文本模型上游可为 OpenAI Chat Completions、Anthropic
  Messages 或 OpenAI Responses（`X-Upstream-Protocol`，默认 `chat`）；客户端与上游
  同协议且无图片时，请求原样透传。
- **上游视觉感知** —— 若上游模型自身声明支持图像（`input_modalities` 含
  `image`），图片将直接透传，不再重复走视觉模型（`X-Upstream-Vision: auto|true|false`）。
- **流式输出** —— SSE 按客户端协议代理/翻译；内部工具调用会缓冲后重放为合法流。
- **不读环境变量** —— 所有凭据、模型名、API 地址均来自 HTTP 请求头。
- **安全** —— 远程图片 SSRF 防护、原子写入、WAL SQLite、脱敏结构化日志、
  路径穿越防护。

## 安装

需要 Python 3.12+。

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## 运行

```bash
python -m llm_visionrelay \
  --host 0.0.0.0 \
  --port 8080 \
  --cache-dir ./data
```

CLI 参数（全部运行时配置来自命令行）：

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--host` / `--port` | 监听地址 / 端口 | `127.0.0.1` / `8080` |
| `--cache-dir` | 缓存目录 | `./data` |
| `--max-image-size` | 单张图片上限（MiB） | `20` |
| `--max-images-per-request` | 单请求图片数上限 | `8` |
| `--max-total-image-bytes` | 单请求图片总字节上限（MiB） | `50` |
| `--timeout` / `--vision-timeout` | 上游 / 视觉模型超时（秒） | `60` / `90` |
| `--vision-max-concurrency` | 每组（地址,Key,模型）并发视觉调用上限 | `8` |
| `--vision-max-retries` | 429/5xx/传输错误的重试次数 | `2` |
| `--management-token` | `/internal/*` 管理令牌（经 `X-Management-Token` 传入） | 无 |

运行测试：

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/python -m pytest
```

## 快速示例

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer TEXT_MODEL_KEY' \
  -H 'X-Upstream-Base-URL: https://text-model.example.com' \
  -H 'X-Vision-Base-URL: https://vision.example.com/v1' \
  -H 'X-Vision-Model: vision-model-name' \
  -H 'X-Vision-Authorization: Bearer VISION_MODEL_KEY' \
  --data-binary @request.json
```

```json
{
  "model": "text-model",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "分析这张图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}
      ]
    }
  ]
}
```

中间件会：提取图片→内容寻址存储→调用视觉模型生成结构化描述→把图片块替换为标注为
**不可信数据**的 `<visual_attachment>` 文本→把纯文本请求转发到
`{X-Upstream-Base-URL}/chat/completions`→以原始协议返回上游响应。

同一张图只会调用一次视觉模型，后续请求命中缓存（`X-Vision-Cache: HIT`）。

## 请求头

| 请求头 | 作用 |
| --- | --- |
| `Authorization` | 文本模型 Key，原样透传上游（绝不发给视觉模型） |
| `X-Upstream-Base-URL` | 文本模型 API 地址（必填） |
| `X-Upstream-Model` | 可选，覆盖请求正文的 `model` |
| `X-Upstream-Protocol` | 上游协议：`chat`（默认）/ `anthropic` / `responses` |
| `X-Upstream-Vision` | `auto`（默认）/ `true` / `false` — 上游支持图像时直接透传 |
| `X-Vision-Base-URL` | 视觉模型地址 |
| `X-Vision-Model` | 视觉模型名 |
| `X-Vision-Authorization` | 视觉模型 Key，只发给视觉接口 |
| `X-Vision-Header-*` | 透传给视觉接口的自定义头 |
| `X-Vision-Auto-Analyze` | 是否自动为每张图生成摘要（默认 `true`） |
| `X-Vision-Tools` | 是否注入 `__vision_` 工具（默认 `true`） |
| `X-Vision-Cache-TTL` | 视觉结果 / URL 别名 TTL（秒，默认 30 天） |
| `X-Vision-Force-Refresh` | 跳过摘要缓存强制重算（默认 `false`） |
| `X-Vision-Cache-Namespace` | 租户命名空间（默认由 `Authorization` 推导） |
| `X-Vision-Params` | 视觉请求的额外 JSON 参数（如思考/推理强度） |
| `X-Request-ID` | 请求 ID，响应中回传（缺省自动生成） |

`X-Vision-Header-*` 有数量与长度限制，且禁止覆盖 `Host`、`Content-Length`、
`Connection`、`Transfer-Encoding`、`Content-Type`、`Authorization`、`Accept`。

## 缓存原理

- 图片内容寻址：`{cache_dir}/objects/sha256/ab/cd/<sha256>`，引用为
  `img_sha256_<64hex>`；相同字节永远映射到同一个 `image_ref`。
- 首次：图片入库，视觉模型生成结构化摘要并写入 SQLite。
- 后续相同图片：复用摘要，不再调用视觉模型。
- 相同 URL：别名有效期内不重新下载；过期后用
  `If-None-Match` / `If-Modified-Since` 条件请求校验内容是否变化。
- 并发相同图片：`singleflight` 去重，只发起一次视觉调用。
- 视觉模型失败时：有（过期）缓存则回退并标记警告，否则返回明确错误，绝不静默丢图。

响应头：

```text
X-Vision-Cache: MISS       # 首次 / 全部未命中
X-Vision-Cache: HIT        # 全部命中
X-Vision-Cache: MIXED      # 部分命中
X-Vision-Image-Refs: img_sha256_xxx,img_sha256_yyy
```

## 内置视觉工具

当 `X-Vision-Tools: true` 且请求含图片时，中间件追加 `__vision_` 工具（自行执行，
不会返回给客户端）：

- `__vision_list_images` —— 列出可用图片引用与缓存摘要（不调用视觉模型）。
- `__vision_analyze` —— 定向分析（query / mode / bbox / force_refresh），带独立缓存。
- `__vision_crop` —— 按归一化区域裁剪，返回新的 `image_ref`。
- `__vision_resize` —— 缩放到目标像素尺寸，返回新的 `image_ref`。
- `__vision_mask` —— 区域蒙版（`blur` / `highlight` / `dim`），返回新的 `image_ref`。

裁剪/缩放/蒙版是真实图像处理（Pillow），结果内容寻址存储并注册到当前租户，模型可
链式操作（例如先裁剪再分析局部细节）。

客户端自定义 tools 会保留；与保留前缀 `__vision_` 冲突返回 400。内部轮次上限 4、
单请求视觉工具调用上限 8，超出后注入错误并禁用内置工具。

## Docker

```bash
docker compose up -d --build
```

服务监听 `http://localhost:8080`，缓存位于 `./data`。构建期代理可通过
`HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` 环境变量提供。

## 管理接口

默认仅允许 loopback；设置 `--management-token` 后需携带 `X-Management-Token`。

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/internal/cache/stats
curl -X DELETE 'http://127.0.0.1:8080/internal/cache?all=true'
curl -X DELETE 'http://127.0.0.1:8080/internal/cache?namespace=client-42'
curl -X DELETE 'http://127.0.0.1:8080/internal/cache?image_ref=img_sha256_<64hex>'
curl -X DELETE 'http://127.0.0.1:8080/internal/cache?expired=true'
```

## 目录结构

```text
llm_visionrelay/
├── app.py               # FastAPI 应用、请求编排、SSE 处理
├── cache_db.py          # SQLite（WAL）持久化缓存
├── cli.py               # CLI 入口
├── config.py            # 进程配置
├── errors.py            # 协议无关错误模型
├── headers.py           # 请求头解析、租户推导
├── image_fetcher.py     # base64 / URL 图片拉取（SSRF 防护）
├── image_store.py       # 内容寻址对象存储（原子写入）
├── imaging.py           # 图片尺寸 / MIME 嗅探
├── imaging_tools.py     # 裁剪 / 缩放 / 蒙版（Pillow）
├── logging.py           # 结构化、脱敏日志
├── message_transform.py # 图片块 → 不可信文本上下文
├── models.py            # Pydantic 请求模型
├── protocols.py         # 客户端协议适配（chat / anthropic / responses）
├── security.py          # 租户 / SSRF / 路径校验
├── tool_loop.py         # 内置视觉工具 + 工具调用循环
├── upstream.py          # 低层上游 HTTP 客户端
├── upstream_models.py   # 上游模型能力注册缓存
├── upstream_protocols.py# 上游协议渲染 / 解析适配
└── vision_client.py     # 视觉模型客户端 + LRU + singleflight
```

## 安全

- 所有 API 凭据、模型名、地址均来自 HTTP 请求头，不读取环境变量或 `.env`。
- 租户隔离基于 SHA-256 摘要，原始 Key 永不落盘或记录。
- 远程图片 SSRF 校验（拒绝 loopback / 链路本地 / 私网 / 保留地址及 DNS rebinding），
  并限制重定向与下载大小。
- 原子写入、WAL SQLite、路径穿越防护。
- 结构化日志脱敏凭据，不记录请求正文或图片内容。

## License

[MIT](LICENSE)
