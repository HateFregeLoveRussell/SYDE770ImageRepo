"""Run the API server directly: python -m app.main --port 6000"""

import argparse

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Tim Hortons Cup Detector API")
    parser.add_argument("--host", default=settings.host, help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=settings.port, help="Port (default: 6000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for dev")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
