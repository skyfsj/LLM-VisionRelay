"""CLI entry point. All runtime settings come from CLI arguments only."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from llm_visionrelay.config import Config
from llm_visionrelay.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-visionrelay",
        description="OpenAI Chat Completions vision middleware proxy",
    )
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="listen port (default: 8080)")
    parser.add_argument("--cache-dir", default="./data", help="cache directory (default: ./data)")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=20,
        help="maximum image size in MiB (default: 20)",
    )
    parser.add_argument(
        "--max-images-per-request",
        type=int,
        default=8,
        help="maximum images per request (default: 8)",
    )
    parser.add_argument(
        "--max-total-image-bytes",
        type=int,
        default=50,
        help="maximum total image bytes per request in MiB (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="default upstream timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--vision-timeout",
        type=float,
        default=90.0,
        help="vision model timeout in seconds (default: 90)",
    )
    parser.add_argument(
        "--vision-max-concurrency",
        type=int,
        default=8,
        help="max concurrent vision requests per (base-url, key, model) group (default: 8)",
    )
    parser.add_argument(
        "--vision-max-retries",
        type=int,
        default=2,
        help="vision retries on 429/5xx/transport errors (default: 2)",
    )
    parser.add_argument(
        "--vision-retry-base-delay",
        type=float,
        default=0.3,
        help="vision retry base delay in seconds (exponential backoff) (default: 0.3)",
    )
    parser.add_argument(
        "--vision-retry-max-delay",
        type=float,
        default=5.0,
        help="vision retry maximum delay in seconds (default: 5)",
    )
    parser.add_argument(
        "--management-token",
        default=None,
        help="optional token required for /internal/* endpoints (passed via X-Management-Token)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="log level (default: INFO)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        host=args.host,
        port=args.port,
        cache_dir=args.cache_dir,
        max_image_bytes=args.max_image_size * 1024 * 1024,
        max_images_per_request=args.max_images_per_request,
        max_total_image_bytes=args.max_total_image_bytes * 1024 * 1024,
        default_timeout=args.timeout,
        vision_timeout=args.vision_timeout,
        vision_max_concurrency=args.vision_max_concurrency,
        vision_max_retries=args.vision_max_retries,
        vision_retry_base_delay=args.vision_retry_base_delay,
        vision_retry_max_delay=args.vision_retry_max_delay,
        management_token=args.management_token,
        log_level=args.log_level,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    import uvicorn

    from llm_visionrelay.app import create_app

    config = config_from_args(args)
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=args.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
