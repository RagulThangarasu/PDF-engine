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
from werkzeug.exceptions import RequestEntityTooLarge

from pdfval import ai_review
from pdfval.cli import run
from pdfval.report.html_report import generate_reports

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(BASE_DIR, "runs")

# A scanned/high-DPI manual pair can run well past 50 MB - that limit was
# hitting real uploads and, worse, showing Werkzeug's raw "Request Entity Too
# Large" page instead of coming back to the form. Overridable via env var for
# a deployment that wants a different ceiling.
MAX_CONTENT_LENGTH = int(os.environ.get("PDFVAL_MAX_UPLOAD_MB", "300")) * 1024 * 1024
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
# run_ids not yet finished, in submission order - lets a queued job report a
# real position ("2 comparisons ahead of you") instead of a flat "waiting".
_QUEUE_ORDER: list[str] = []
# The validation report, the side-by-side section browser, the chapter
# comparison (with its page renders under chapters/), the full-PDF viewer
# (pdf.html, pages under pdfview/) and the TOC comparison are
# exposed. report.json is still written (the result page reads it server-side)
# but is not downloadable; report.pdf is not produced at all.
_DOWNLOAD_NAME_RE = re.compile(
    r"^((report|sections|toc|chapters|pdf)\.html|issues\.pdf|(screenshots|sections|chapters|pdfview)/[\w\-.]+\.(png|jpg))$"
)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.secret_key = os.environ.get("PDFVAL_SECRET_KEY") or secrets.token_hex(32)

    os.makedirs(RUNS_DIR, exist_ok=True)

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_exc):
        # Without this, an over-limit upload hit Werkzeug's bare "Request
        # Entity Too Large" page - no styling, no way back to the form.
        limit_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
        flash(f"That upload is too large — combined PDF size must be under {limit_mb} MB.")
        return redirect(url_for("upload_form"))

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
        ai_enabled = request.form.get("ai_review") == "on"
        if ai_enabled:
            open(os.path.join(run_dir, "ai_enabled"), "w").close()
        _start_job(app, run_id, run_dir, expected_path, actual_path, exp_name, act_name, ai_enabled)
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
            ai_enabled = os.path.isfile(os.path.join(run_dir, "ai_enabled"))
            _start_job(app, run_id, run_dir, expected_path, actual_path, *names, ai_enabled)
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
        categories, total = _category_counts(data)
        ai_notes = None
        try:
            with open(os.path.join(RUNS_DIR, run_id, "ai_review.json"), "r", encoding="utf-8") as f:
                ai_notes = json.load(f)
        except FileNotFoundError:
            pass
        except Exception:
            ai_notes = None
        return render_template("result.html", run_id=run_id, report=data, categories=categories, total=total,
                              ai_notes=ai_notes)

    @app.get("/runs/<run_id>/<path:filename>")
    def download(run_id: str, filename: str):
        if not _RUN_ID_RE.match(run_id) or not _DOWNLOAD_NAME_RE.match(filename):
            abort(404)
        # A rerun rewrites pdf.html / report.json in place, same file name -
        # without this a browser that already cached the old copy (or an
        # in-flight "is this stale?" check that never lands) keeps showing the
        # pre-rerun issues, which reads as the fix not having worked at all.
        response = send_from_directory(os.path.join(RUNS_DIR, run_id), filename)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    return app


def _category_counts(data: dict) -> tuple[list[dict] | None, int]:
    """(issues per category, total) from report.json - the same categories
    pdf.html's nav files each issue under, so the two agree. None for a run made
    before issues carried a category, whose result page lists its checks instead.
    """
    from pdfval.validators.chapter import CATEGORIES

    counts = {c["label"]: 0 for c in CATEGORIES}
    total = uncategorised = 0
    for check in data.get("checks") or []:
        for issue in check.get("issues") or []:
            total += 1
            label = (issue.get("details") or {}).get("category")
            if label in counts:
                counts[label] += 1
            else:
                uncategorised += 1
    if uncategorised and uncategorised == total:
        return None, total
    return [{"label": c["label"], "help": c["help"], "count": counts[c["label"]]} for c in CATEGORIES], total


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


def _queue_position(run_id: str) -> int:
    """1-based position in the queue; 1 = running now or about to start."""
    with _JOBS_LOCK:
        try:
            return _QUEUE_ORDER.index(run_id) + 1
        except ValueError:
            return 1


def _run_ai_review(run_dir: str, run_id: str, expected_path: str, actual_path: str) -> None:
    """The optional AI pass - runs after the deterministic report is already
    written, and can never fail the run: any error here (Ollama not running,
    model missing, a page timing out) is swallowed and simply leaves no
    ai_review.json, so the page just doesn't show that section.
    """
    try:
        import fitz
        if not ai_review.available():
            return
        exp_pages = fitz.open(expected_path).page_count
        act_pages = fitz.open(actual_path).page_count
        page_count = min(exp_pages, act_pages)
        pdfview_dir = os.path.join(run_dir, "pdfview")

        def cb(i: int, n: int) -> None:
            _set_progress(run_dir, run_id, percent=98, label=f"AI visual review — page {i}/{n}")

        _set_progress(run_dir, run_id, percent=98, label="AI visual review starting…")
        notes = ai_review.review_pages(pdfview_dir, page_count, progress_cb=cb)
        with open(os.path.join(run_dir, "ai_review.json"), "w", encoding="utf-8") as f:
            json.dump({"model": ai_review.VISION_MODEL, "notes": notes}, f)
    except Exception:  # noqa: BLE001
        traceback.print_exc()


_QUEUE_POLL_SECONDS = 3.0


def _start_job(app, run_id, run_dir, expected_path, actual_path, exp_name, act_name,
               ai_enabled: bool = False) -> None:
    with _JOBS_LOCK:
        _QUEUE_ORDER.append(run_id)
    _set_progress(run_dir, run_id, percent=1, label="Starting…", done=False, error=None,
                  result_url=f"/runs/{run_id}")

    def worker():
        try:
            # Poll for the lock instead of blocking on it outright, so a
            # queued job's progress keeps reporting how many comparisons are
            # still ahead of it rather than sitting on one static message.
            while not _RUN_LOCK.acquire(timeout=_QUEUE_POLL_SECONDS):
                ahead = _queue_position(run_id) - 1
                if ahead <= 0:
                    continue  # lock lost a race right as it freed up - retry
                label = (
                    "Waiting for another comparison to finish…" if ahead == 1
                    else f"Waiting — {ahead} comparisons ahead of you…"
                )
                _set_progress(run_dir, run_id, percent=1, label=label)
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
                    report, run_dir, formats=("json", "pdfview", "toc"),
                    new_comparison_url=url_for("upload_form"),
                )
            if ai_enabled:
                _run_ai_review(run_dir, run_id, expected_path, actual_path)
            _set_progress(run_dir, run_id, percent=100, label="Done", done=True)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _set_progress(run_dir, run_id, percent=100, label="Failed", done=True,
                          error=f"{type(exc).__name__}: {exc}")
        finally:
            with _JOBS_LOCK:
                if run_id in _QUEUE_ORDER:
                    _QUEUE_ORDER.remove(run_id)

    threading.Thread(target=worker, name=f"pdfval-{run_id[:8]}", daemon=True).start()
