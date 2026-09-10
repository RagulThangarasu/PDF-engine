"""Flask web UI: upload Production/Staging PDFs, run validation, serve reports."""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
import traceback
import uuid

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

from pdfval.cli import run
from pdfval.report.html_report import generate_reports

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(BASE_DIR, "runs")

MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB per request
PDF_MAGIC = b"%PDF-"

_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# In-process registry of running / finished comparison jobs, keyed by run_id.
# The dev server is single-process so this is enough; a copy of the live
# progress is also written to <run_dir>/progress.json as a fallback.
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
# pdfplumber / extractor caches are module-global, so only one comparison runs
# at a time; others queue behind this lock.
_RUN_LOCK = threading.Lock()
# The validation report and the side-by-side section browser are exposed.
# report.json is still written (the result page reads it server-side) but is
# not downloadable, and report.pdf is not produced for web runs at all.
_DOWNLOAD_NAME_RE = re.compile(
    r"^((report|sections|toc)\.html|(screenshots|sections)/[\w\-.]+\.(png|jpg))$"
)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.secret_key = os.environ.get("PDFVAL_SECRET_KEY") or secrets.token_hex(32)

    os.makedirs(RUNS_DIR, exist_ok=True)

    @app.get("/")
    def upload_form():
        token = secrets.token_hex(16)
        session["csrf_token"] = token
        return render_template("upload.html", csrf_token=token)

    @app.post("/compare")
    def compare():
        if request.form.get("csrf_token") != session.pop("csrf_token", None):
            flash("Your session expired, please try again.")
            return redirect(url_for("upload_form"))

        expected_file = request.files.get("expected")
        actual_file = request.files.get("actual")

        if not _is_valid_pdf(expected_file):
            flash("Production PDF is missing or is not a valid PDF file.")
            return redirect(url_for("upload_form"))
        if not _is_valid_pdf(actual_file):
            flash("Staging PDF is missing or is not a valid PDF file.")
            return redirect(url_for("upload_form"))

        run_id = uuid.uuid4().hex
        run_dir = os.path.join(RUNS_DIR, run_id)
        uploads_dir = os.path.join(run_dir, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        expected_path = os.path.join(uploads_dir, "expected.pdf")
        actual_path = os.path.join(uploads_dir, "actual.pdf")
        expected_file.save(expected_path)
        actual_file.save(actual_path)

        exp_name = _display_name(expected_file.filename, "expected.pdf")
        act_name = _display_name(actual_file.filename, "actual.pdf")
        _start_job(app, run_id, run_dir, expected_path, actual_path, exp_name, act_name)
        return redirect(url_for("processing", run_id=run_id))

    @app.post("/runs/<run_id>/rerun")
    def rerun(run_id: str):
        """Re-run the whole comparison against the PDFs already uploaded for
        this run - the "the matching looks off, do it again" button."""
        if not _RUN_ID_RE.match(run_id):
            abort(404)
        run_dir = os.path.join(RUNS_DIR, run_id)
        expected_path = os.path.join(run_dir, "uploads", "expected.pdf")
        actual_path = os.path.join(run_dir, "uploads", "actual.pdf")
        if not (os.path.isfile(expected_path) and os.path.isfile(actual_path)):
            abort(404)
        with _JOBS_LOCK:
            busy = run_id in _JOBS and not _JOBS[run_id].get("done")
        if not busy:
            names = _run_names(run_dir)
            _start_job(app, run_id, run_dir, expected_path, actual_path, *names)
        return redirect(url_for("processing", run_id=run_id))

    @app.get("/runs/<run_id>/processing")
    def processing(run_id: str):
        if not _RUN_ID_RE.match(run_id):
            abort(404)
        st = _job_status(run_id)
        if st.get("done") and not st.get("error"):
            return redirect(url_for("result", run_id=run_id))
        return render_template("processing.html", run_id=run_id, status=st)

    @app.get("/runs/<run_id>/status")
    def status(run_id: str):
        if not _RUN_ID_RE.match(run_id):
            abort(404)
        return jsonify(_job_status(run_id))

    @app.get("/runs/<run_id>")
    def result(run_id: str):
        if not _RUN_ID_RE.match(run_id):
            abort(404)
        json_path = os.path.join(RUNS_DIR, run_id, "report.json")
        if not os.path.isfile(json_path):
            st = _job_status(run_id)
            if st and not st.get("done"):
                return redirect(url_for("processing", run_id=run_id))
            abort(404)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return render_template("result.html", run_id=run_id, report=data)

    @app.get("/runs/<run_id>/<path:filename>")
    def download(run_id: str, filename: str):
        if not _RUN_ID_RE.match(run_id) or not _DOWNLOAD_NAME_RE.match(filename):
            abort(404)
        return send_from_directory(os.path.join(RUNS_DIR, run_id), filename)

    return app


def _is_valid_pdf(file_storage) -> bool:
    if not file_storage or not file_storage.filename:
        return False
    if not file_storage.filename.lower().endswith(".pdf"):
        return False
    header = file_storage.stream.read(len(PDF_MAGIC))
    file_storage.stream.seek(0)
    return header == PDF_MAGIC


def _display_name(filename: str | None, fallback: str) -> str:
    if not filename:
        return fallback
    return os.path.basename(filename)[:255] or fallback


def _run_names(run_dir: str) -> tuple[str, str]:
    """Recover the display names used for a previous run (for a re-run)."""
    try:
        with open(os.path.join(run_dir, "report.json"), "r", encoding="utf-8") as f:
            d = json.load(f)
        return (d.get("expected_path") or "expected.pdf", d.get("actual_path") or "actual.pdf")
    except Exception:
        return ("expected.pdf", "actual.pdf")


def _set_progress(run_dir: str, run_id: str, **fields) -> None:
    with _JOBS_LOCK:
        job = _JOBS.setdefault(run_id, {})
        job.update(fields)
        job["ts"] = time.time()
        snapshot = dict(job)
    try:
        with open(os.path.join(run_dir, "progress.json"), "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
    except OSError:
        pass


def _job_status(run_id: str) -> dict:
    with _JOBS_LOCK:
        if run_id in _JOBS:
            return dict(_JOBS[run_id])
    try:
        with open(os.path.join(RUNS_DIR, run_id, "progress.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        if os.path.isfile(os.path.join(RUNS_DIR, run_id, "report.json")):
            return {"percent": 100, "label": "Done", "done": True}
        return {}


def _start_job(app, run_id, run_dir, expected_path, actual_path, exp_name, act_name) -> None:
    _set_progress(run_dir, run_id, percent=1, label="Starting…", done=False, error=None,
                  result_url=f"/runs/{run_id}")

    def worker():
        try:
            if not _RUN_LOCK.acquire(blocking=False):
                _set_progress(run_dir, run_id, percent=1, label="Waiting for another comparison to finish…")
                _RUN_LOCK.acquire()
            try:
                report = run(
                    expected_path, actual_path, run_dir,
                    progress_cb=lambda pct, label: _set_progress(run_dir, run_id, percent=round(pct), label=label),
                )
            finally:
                _RUN_LOCK.release()
            report.expected_path = exp_name
            report.actual_path = act_name
            _set_progress(run_dir, run_id, percent=97, label="Writing report files")
            with app.test_request_context():
                generate_reports(
                    report, run_dir, formats=("json", "html", "sections", "toc"),
                    new_comparison_url=url_for("upload_form"),
                )
            _set_progress(run_dir, run_id, percent=100, label="Done", done=True)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _set_progress(run_dir, run_id, percent=100, label="Failed", done=True,
                          error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=worker, name=f"pdfval-{run_id[:8]}", daemon=True).start()
