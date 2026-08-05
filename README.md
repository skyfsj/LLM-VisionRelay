# LLM-VisionRelay

> English | [中文](README_CN.md)

A Python async HTTP middleware that makes any **text-only** chat model vision-capable.

`llm-visionrelay` accepts images from OpenAI Chat Completions, Anthropic Messages, or
OpenAI Responses clients, runs them through a **vision model** to produce a structured
text description, replaces the image blocks with that description, and forwards the
result to your text model — all while speaking the client's original protocol.

## Features

- **Multi-protocol input** — auto-detected by path:
  - `POST /v1/chat/completions` (OpenAI Chat Completions)
  - `POST /v1/messages` (Anthropic Messages)
  - `POST /v1/responses` (OpenAI Responses)
- **Vision extraction pipeline** — images are fetched (base64 data URLs, `http(s)://`),
  stored in a **content-addressed cache**, and described by the vision model in a
  native-multimodal style: a dense natural-language `description` plus structured
  elements (`ocr`, `objects`, `layout`, `relationships`) each with **normalized and
  pixel bounding boxes** (`bbox` / `bbox_px`). The image is then replaced with that
  context before being sent to the text model.
- **Multi-layer caching** — SQLite metadata + content-addressed file store + in-process
  LRU + `singleflight` concurrency dedup. The same image is only sent to the vision model
  once; the same URL is only re-downloaded when its alias expires.
- **Tenant isolation** — cache keys use the SHA-256 of the `Authorization` header (or an
  explicit `X-Vision-Cache-Namespace`); raw keys are never stored. Tenants cannot read
  each other's images or results.
- **Built-in vision tools** — the middleware injects `__vision_` tools so the text model
  can request further analysis, and executes them itself:
  - `__vision_list_images`, `__vision_analyze`
  - `__vision_crop`, `__vision_resize`, `__vision_mask` (real image processing via Pillow)
- **Upstream protocol support** — the text model upstream can be OpenAI Chat Completions,
  Anthropic Messages, or OpenAI Responses (`X-Upstream-Protocol`, default `chat`); when
  the client and upstream share a protocol with no images, requests are proxied verbatim.
- **Upstream vision detection** — if the upstream model itself declares image input
  (`input_modalities` includes `image`), images are passed through untouched instead of
  being re-processed by the vision model (`X-Upstream-Vision: auto|true|false`).
- **Streaming** — SSE is proxied/translated in the client's protocol; internal tool calls
  are buffered and re-emitted as a valid stream.
- **No environment-variable config** — all credentials, model names, and API addresses
  come from HTTP request headers; nothing is read from the environment.
- **Security** — SSRF protection for remote images, atomic writes, WAL SQLite, redacting
  structured logs, path-traversal protection.

## Install

Requires Python 3.12+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Run

```bash
python -m llm_visionrelay \
  --host 0.0.0.0 \
  --port 8080 \
  --cache-dir ./data
```

CLI options (all runtime settings come from the command line):

| Option | Description | Default |
| --- | --- | --- |
| `--host` / `--port` | listen address / port | `127.0.0.1` / `8080` |
| `--cache-dir` | cache directory | `./data` |
| `--max-image-size` | max single image size (MiB) | `20` |
| `--max-images-per-request` | max images per request | `8` |
| `--max-total-image-bytes` | max total image bytes per request (MiB) | `50` |
| `--timeout` | upstream text model timeout (s) | `60` |
| `--vision-max-concurrency` | max concurrent vision calls per (base-url, key, model) group | `8` |
| `--vision-max-retries` | vision retries on 429/5xx/transport errors | `2` |
| `--management-token` | optional token for `/internal/*` endpoints (sent via `X-Management-Token`) | none |

Run the tests:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/python -m pytest
```

## Quick example

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
        {"type": "text", "text": "Analyze this diagram"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}
      ]
    }
  ]
}
```

The middleware:
1. Extracts the image, stores it content-addressed, and asks the vision model for a
   structured description.
2. Replaces the image block with the description, wrapped in an explicit
   `<visual_attachment>` marker flagged as **untrusted data**.
3. Forwards the text-only request to `{X-Upstream-Base-URL}/chat/completions` with your
   `Authorization`.
4. Returns the upstream response in the original protocol.

Caching means the same image is only sent to the vision model **once** — subsequent
requests hit the cache (`X-Vision-Cache: HIT`).

## Request headers

| Header | Purpose |
| --- | --- |
| `Authorization` | text-model key, forwarded verbatim to the upstream (never sent to the vision model) |
| `X-Upstream-Base-URL` | text-model API base URL (required) |
| `X-Upstream-Model` | optional override for the request `model` |
| `X-Upstream-Protocol` | upstream protocol: `chat` (default) / `anthropic` / `responses` |
| `X-Upstream-Vision` | `auto` (default) / `true` / `false` — pass images through when the upstream is vision-capable |
| `X-Vision-Base-URL` | vision model base URL |
| `X-Vision-Model` | vision model name |
| `X-Vision-Authorization` | vision model key, sent only to the vision endpoint (bare keys like `sk-...` are auto-prefixed with `Bearer ` for standard APIs) |
| `X-Vision-Header-*` | extra headers forwarded to the vision endpoint |
| `X-Vision-Auto-Analyze` | auto-generate a summary for each image (default `true`) |
| `X-Vision-Tools` | inject `__vision_` tools (default `true`) |
| `X-Vision-Cache-TTL` | vision result / URL alias TTL in seconds (default 30 days) |
| `X-Vision-Force-Refresh` | bypass the summary cache (default `false`) |
| `X-Vision-Cache-Namespace` | tenant namespace (default: derived from `Authorization`) |
| `X-Vision-Params` | extra JSON body params for the vision request (e.g. thinking / reasoning effort) |
| `X-Vision-Reasoning` | vision-model thinking: `on` / `off` / `auto` (default `auto`). `off` sends `thinking: disabled` |
| `X-Vision-Reasoning-Effort` | override the vision reasoning level: `none` / `low` / `medium` / `high` / `max` / `auto` (default `auto`; falls back to the next lower supported level) |
| `X-Vision-Reasoning-Budget` | cap the vision chain-of-thought tokens (0 disables), overrides `--vision-reasoning-budget`; keeps thinking from starving the answer (default 2048, clamped to half of `max_tokens`) |
| `X-Vision-Max-Tokens` | cap the vision model's output tokens (1–200000), overrides `--vision-max-tokens`; prevents chain-of-thought runaway |
| `X-Vision-Max-Images` | per-request image count cap (1–4096), overrides `--max-images-per-request` |
| `X-Vision-Max-Image-Bytes` | per-request single-image size cap in MiB (1–200) |
| `X-Vision-Max-Total-Image-Bytes` | per-request total image bytes cap in MiB (1–2048) |
| `X-Request-ID` | request id echoed on the response (generated if absent) |

`X-Vision-Header-*` headers have count and length limits, and cannot override
`Host`, `Content-Length`, `Connection`, `Transfer-Encoding`, `Content-Type`,
`Authorization`, or `Accept`.

## Protocol parameter passthrough

Generation parameters are translated between client and upstream protocols so
they are not lost when protocols differ:

| Parameter | Chat Completions | Anthropic Messages | OpenAI Responses |
| --- | --- | --- | --- |
| output cap | `max_tokens` / `max_completion_tokens` | `max_tokens` | `max_output_tokens` |
| sampling | `temperature`, `top_p` | `temperature`, `top_p` | `temperature`, `top_p` |
| stop | `stop` | `stop_sequences` | — |
| tool choice | `tool_choice` | `tool_choice` (`auto/any/tool`) | `tool_choice` |
| reasoning | `reasoning_effort` | `thinking` (budget) | `reasoning: {effort}` |
| metadata | `metadata` | `metadata` | `metadata` |
| parallel tools | `parallel_tool_calls` | — | `parallel_tool_calls` |
| user / store | `user` | — | `user`, `store` |
| structured output | `response_format` | — | `text: {format}` |

Chat requests also preserve unknown extension fields verbatim. Streamed responses
carry the same mappings (e.g. `reasoning_content` becomes an Anthropic `thinking`
block or a Responses `reasoning` output item).

The vision model is called with the same reasoning intensity the agent requested
(`reasoning_effort` / `reasoning.effort`), overridable per request via
`X-Vision-Reasoning` / `X-Vision-Reasoning-Effort`. If the vision model does not
support a level that high, the middleware automatically falls back to the next
lower supported level (supported levels are configurable via
`vision_reasoning_levels`, default `low`/`medium`/`high`). Reasoning level and
thinking toggle are part of the vision cache key, so different intensities never
reuse each other's analysis. Vision output tokens are capped (`vision_max_tokens`,
default 8192) so a small reasoning model cannot loop its chain-of-thought
forever, and the chain-of-thought budget is capped (`vision_reasoning_budget`,
default 2048, clamped to half of `max_tokens`) so thinking can never eat the
whole output and leave an empty answer. No timeout is imposed on the vision model
— only the client agent's own disconnect/interrupt stops it.

## How caching works

- Images are content-addressed: `{cache_dir}/objects/sha256/ab/cd/<sha256>`, referenced as
  `img_sha256_<64-hex>`. Identical bytes always map to the same `image_ref`.
- First time: the image is stored and the vision model produces a structured summary,
  persisted in SQLite.
- Later requests with the same image reuse the summary — no vision call.
- Same URL: not re-downloaded while the URL alias is valid; when it expires, a conditional
  request (`If-None-Match` / `If-Modified-Since`) checks whether the content changed.
- If a client disconnects mid-analysis, the summary batch keeps running in the
  background and caches the finished results, so a resumed session never re-reads
  the images.
- Concurrent requests for the same image are deduplicated via `singleflight`.
- If the vision model fails and an expired cache entry exists, it is used with a warning
  marker rather than failing the request.

Response headers:

```text
X-Vision-Cache: MISS       # first / all misses
X-Vision-Cache: HIT        # all hits
X-Vision-Cache: MIXED      # some hits, some misses
X-Vision-Image-Refs: img_sha256_xxx,img_sha256_yyy
```

## Built-in vision tools

When `X-Vision-Tools: true` and the request contains images, the middleware appends
`__vision_` tools (executed internally, never returned to the client):

- `__vision_list_images` — list available image refs and cached summaries (no vision call).
- `__vision_analyze` — targeted analysis of an image (query / mode / bbox / force_refresh),
  with its own result cache.
- `__vision_crop` — crop an image to a normalized region, returning a new `image_ref`.
- `__vision_resize` — resize an image to a target pixel size, returning a new `image_ref`.
- `__vision_mask` — mask a region (`blur` / `highlight` / `dim`), returning a new `image_ref`.

Crop/resize/mask results are real image processing (Pillow), stored content-addressed and
registered to the current tenant, so the model can chain operations (e.g. crop then analyze
a detail region).

Client-defined tools are preserved; names colliding with the reserved `__vision_` prefix are
rejected with HTTP 400. Internal rounds are capped at 4 and vision tool calls at 8 per
request; beyond that an error is injected and internal tools are disabled.

## Docker

```bash
docker compose up -d --build
```

Serves `http://localhost:8080`. The cache lives in `./data`. Build-time proxy settings can
be supplied via `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` environment variables.

## Management endpoints

By default only reachable from loopback; with `--management-token` set, requests must send
`X-Management-Token`.

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/internal/cache/stats
curl -X DELETE 'http://127.0.0.1:8080/internal/cache?all=true'
curl -X DELETE 'http://127.0.0.1:8080/internal/cache?namespace=client-42'
curl -X DELETE 'http://127.0.0.1:8080/internal/cache?image_ref=img_sha256_<64hex>'
curl -X DELETE 'http://127.0.0.1:8080/internal/cache?expired=true'
```

## Architecture

```text
llm_visionrelay/
├── app.py               # FastAPI app, request orchestration, SSE handling
├── cache_db.py          # SQLite (WAL) persistent cache
├── cli.py               # CLI entry point
├── config.py            # process configuration
├── errors.py            # protocol-agnostic error model
├── headers.py           # request-header parsing, tenant derivation
├── image_fetcher.py     # base64 / URL image fetching (SSRF-guarded)
├── image_store.py       # content-addressed object store (atomic writes)
├── imaging.py           # image dimension / MIME sniffing
├── imaging_tools.py     # crop / resize / mask (Pillow)
├── logging.py           # structured, redacting logs
├── message_transform.py # image blocks → untrusted text context
├── models.py            # Pydantic request models
├── protocols.py         # client protocol adapters (chat / anthropic / responses)
├── security.py          # tenant / SSRF / path validation
├── tool_loop.py         # built-in vision tools + tool-call loop
├── upstream.py          # low-level upstream HTTP client
├── upstream_models.py   # cached upstream model capability registry
├── upstream_protocols.py# upstream protocol render / parse adapters
└── vision_client.py     # vision model client + LRU + singleflight
```

## Security

- All API credentials, model names, and addresses are supplied via HTTP headers; no
  environment-variable or `.env` config loading.
- Tenant isolation via SHA-256 digests; raw keys are never persisted or logged.
- Remote images are SSRF-checked (loopback / link-local / private / reserved networks and
  DNS-rebinding resolution are rejected), with redirect and download-size limits.
- Atomic file writes, WAL SQLite, and path-traversal protection.
- Structured logs redact credentials and never record request bodies or image payloads.

## License

[MIT](LICENSE)
