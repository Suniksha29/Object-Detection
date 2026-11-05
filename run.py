"""CLI entrypoint to launch the Object Detector & Counter web server."""

import argparse
import os
import uvicorn


def main():
    default_host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    default_port = int(os.environ.get("PORT", 8000))

    parser = argparse.ArgumentParser(description="Run Object Detector & Counter Server")
    parser.add_argument("--host", type=str, default=default_host, help=f"Host interface to bind to (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to listen on (default: {default_port})")
    parser.add_argument("--reload", action="store_true", help="Enable automatic reload on code changes")
    args = parser.parse_args()

    print(f"Starting Object Detector & Counter at http://{args.host}:{args.port}")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

