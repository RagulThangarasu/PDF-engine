"""Launch the PDF Validation Engine web UI: `python -m pdfval.web`."""
from __future__ import annotations

import argparse
import os
import socket
import sys

from pdfval.web.app import create_app

DEFAULT_PORT = 5001


def port_owner(host: str, port: int) -> str:
    """Who already answers on this port, or "" when nobody does.

    Checked before binding so a port someone else owns is reported as such,
    rather than as a 403 from a stranger halfway through a review.
    """
    for family, addr in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.4)
                if probe.connect_ex((addr, port)) == 0:
                    return addr
        except OSError:
            continue
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdfval.web", description="Run the PDF Validation Engine web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    # NOT 5000. macOS runs AirPlay Receiver (ControlCenter) on *:5000, which
    # answers every HTTP request with 403 "Access denied". Flask binding
    # 127.0.0.1:5000 wins for that exact address, but `localhost` resolves to
    # ::1 first on macOS and AirPlay answers there - so the app loads from one
    # spelling of the address and is "denied" from the other, which looks like
    # a permissions bug in the app and is not one.
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode (local development only)")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Do not restart the server when the source changes (reloading is on by default)",
    )
    args = parser.parse_args(argv)

    # Only in the process that actually binds. The reloader restarts this
    # module in a child while the PARENT still holds the socket, so a check
    # that runs in both sees the server's own port as taken and kills it on
    # the first restart - the server comes up, then dies the moment a file
    # changes.
    taken = "" if os.environ.get("WERKZEUG_RUN_MAIN") else port_owner(args.host, args.port)
    if taken:
        print(f"Something is already listening on port {args.port} ({taken}).", file=sys.stderr)
        if args.port == 5000:
            print("On macOS that is usually AirPlay Receiver, which answers HTTP with "
                  "403 'Access denied'.\nTurn it off in System Settings > General > "
                  "AirDrop & Handoff, or use another port:", file=sys.stderr)
        print(f"    python -m pdfval.web --port {args.port + 1}", file=sys.stderr)
        return 1

    app = create_app()
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        print(f"\n  PDF Validation Engine: http://{args.host}:{args.port}\n")
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
