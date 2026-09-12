"""Launch the PDF Validation Engine web UI: `python -m pdfval.web`."""
from __future__ import annotations

import argparse

from pdfval.web.app import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdfval.web", description="Run the PDF Validation Engine web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode (local development only)")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Do not restart the server when the source changes (reloading is on by default)",
    )
    args = parser.parse_args(argv)

    app = create_app()
    # Reload on source changes by DEFAULT. Jinja re-reads its templates from
    # disk on every render but Python modules are imported once, so a
    # long-running server quietly serves half-new output after an edit - new
    # report.html, but the old section browser, which came out empty. That is
    # indistinguishable from a bug in the report itself, so the server picks up
    # code changes unless told not to. (`use_reloader` without `debug` keeps
    # the interactive debugger - and its remote code execution - switched off.)
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=not args.no_reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
