#!/usr/bin/env python3
"""
Run the Call Analysis System Web Application
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config  # type: ignore


def main() -> None:
    """
    Entry point for running the FastAPI-based Call Analysis web application.

    Uses uvicorn as the ASGI server.
    """
    import uvicorn

    host = Config.HOST
    port = Config.PORT

    print("Starting Call Analysis Web Application (FastAPI)...")
    print("=" * 50)
    print(f"Access the API at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        # Use module path so uvicorn reload works if enabled
        uvicorn.run(
            "call_analysis.web_app_fastapi:app",
            host=host,
            port=port,
            reload=Config.DEBUG,
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting web application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

