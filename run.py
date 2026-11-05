"""CLI entrypoint to launch the Object Detector & Counter web server."""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Object Detector & Counter Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable automatic reload on code changes")
    args = parser.parse_args()

    print(f"Starting Object Detector & Counter at http://{args.host}:{args.port}")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

