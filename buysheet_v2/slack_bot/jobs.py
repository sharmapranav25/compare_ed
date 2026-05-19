"""Background job processor.

One worker thread, FIFO queue. Each job is one PDF upload from Slack:
  1. download the PDF from Slack into a tempdir
  2. ack the user in the upload's thread with page count + ETA
  3. run the pipeline with a progress callback that pings 25/50/75%
  4. write the workbook (text-only by default — image binding is v2)
  5. upload the workbook back to the thread with a summary

Failures inside one job log + post an error reply but do NOT crash the worker;
the next queued job runs as normal.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from buysheet_v2.pipeline import run_pipeline
from buysheet_v2.slack_bot import formatting
from buysheet_v2.verify import oracle_summary
from buysheet_v2.write import (
    CONTRADICTED_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    write_workbook,
)

log = logging.getLogger(__name__)

# Slack file downloads need the bot token in the Authorization header — the
# url_private link is not publicly accessible.
_REQUEST_TIMEOUT_SEC = 60
_MAX_PDF_BYTES = 50 * 1024 * 1024  # Slack file API cap

# Persistent runs directory: every job keeps its PDF + extraction sidecar +
# output xlsx here so we can do post-hoc accuracy/discrepancy analysis after
# a batch of test uploads. Overridable via env so a future deployment can
# pin it to a mounted volume.
RUNS_ROOT = Path(
    os.environ.get("BUYSHEET_RUNS_DIR", str(Path.home() / "buysheet_runs"))
).resolve()


def _run_dir_for(file_name: str) -> Path:
    """Create a fresh run directory: ~/buysheet_runs/<stem>/<YYYYMMDD-HHMMSS>/.

    Keyed by file stem (not vendor key, since vendor key isn't known until
    extraction completes) and timestamped so re-uploads of the same PDF get
    their own folders.
    """
    safe_stem = Path(file_name).stem.lower().replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RUNS_ROOT / safe_stem / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class JobRequest:
    """One PDF-upload job pulled from Slack."""
    file_id: str
    file_name: str
    file_url_private: str
    channel: str
    thread_ts: str  # post all replies in this thread
    user_id: str


class JobQueue:
    """Single-worker FIFO queue. Survives transient errors in any job."""

    def __init__(self, slack_client: WebClient, bot_token: str):
        self._client = slack_client
        self._bot_token = bot_token
        self._q: queue.SimpleQueue[JobRequest] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        # Snapshot of queue depth at enqueue time so the ack message can say
        # "queued behind N other jobs." SimpleQueue has no len(), so we track
        # it ourselves.
        self._pending_count = 0

    def enqueue(self, job: JobRequest) -> int:
        """Add a job. Returns queue position (0 = will start immediately)."""
        with self._lock:
            position = self._pending_count
            self._pending_count += 1
            self._ensure_worker_locked()
        self._q.put(job)
        return position

    def _ensure_worker_locked(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._run, name="buysheet-job-worker", daemon=True,
            )
            self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                job = self._q.get(timeout=30 * 60)
            except queue.Empty:
                # Idle for 30 minutes — exit the worker; the next enqueue
                # spawns a fresh one.
                return
            try:
                _process_job(self._client, self._bot_token, job)
            except Exception as exc:
                log.exception("Unhandled error in job %s", job.file_name)
                _safe_post(
                    self._client, job,
                    formatting.error_message(
                        job.file_name, type(exc).__name__, str(exc)[:200],
                    ),
                )
            finally:
                with self._lock:
                    self._pending_count = max(0, self._pending_count - 1)


def _process_job(client: WebClient, bot_token: str, job: JobRequest) -> None:
    log.info("[%s] starting job from user=%s", job.file_name, job.user_id)
    run_dir = _run_dir_for(job.file_name)
    log.info("[%s] persisting artefacts to %s", job.file_name, run_dir)

    # Save the source PDF + a small request manifest first; everything else
    # (sidecar, xlsx) lands alongside it as the pipeline progresses.
    pdf_path = run_dir / job.file_name
    _download_to(job.file_url_private, bot_token, pdf_path)
    (run_dir / "request.json").write_text(
        '{"slack_user_id": "%s", "slack_channel": "%s", "slack_thread_ts": "%s", '
        '"slack_file_id": "%s", "file_name": "%s"}'
        % (job.user_id, job.channel, job.thread_ts, job.file_id, job.file_name)
    )

    # Count pages cheaply for the ack message + ETA.
    try:
        with pdfium.PdfDocument(str(pdf_path)) as pdf:
            page_count = len(pdf)
    except Exception as exc:
        _safe_post(
            client, job,
            formatting.error_message(
                job.file_name, type(exc).__name__,
                f"Could not open as a PDF: {exc}",
            ),
        )
        return

    _safe_post(client, job, formatting.ack_message(job.file_name, page_count))

    progress_cb = _make_progress_callback(client, job)
    try:
        # run_pipeline writes <pdf_stem>.v2.cards.json next to the PDF, so
        # passing the persistent path means the sidecar persists for analysis.
        extraction = run_pipeline(
            pdf_path,
            verbose=False,             # quiet logs — Slack is the UI
            progress_callback=progress_cb,
        )
    except Exception as exc:
        tb = traceback.format_exc(limit=2).splitlines()[-1]
        _safe_post(
            client, job,
            formatting.error_message(
                job.file_name, type(exc).__name__, tb[:200],
            ),
        )
        return

    out_path = run_dir / f"BUYSHEET_{extraction.vendor_key}_v2.xlsx"
    try:
        write_stats = write_workbook(
            extraction, out_path,
            pdf_path=pdf_path,
            embed_photos=True,
        )
    except Exception as exc:
        tb = traceback.format_exc(limit=2).splitlines()[-1]
        _safe_post(
            client, job,
            formatting.error_message(
                job.file_name, type(exc).__name__,
                f"Extraction succeeded but workbook write failed: {tb[:160]}",
            ),
        )
        return

    summary = _build_summary(extraction, write_stats)
    _safe_post(
        client, job,
        formatting.done_message(job.file_name, **summary),
    )
    _upload_file(client, job, out_path)
    log.info("[%s] complete — artefacts in %s", job.file_name, run_dir)


def _download_to(file_url_private: str, bot_token: str, dest: Path) -> None:
    """Download a Slack file. url_private requires Bearer auth."""
    headers = {"Authorization": f"Bearer {bot_token}"}
    with requests.get(
        file_url_private, headers=headers, stream=True,
        timeout=_REQUEST_TIMEOUT_SEC,
    ) as resp:
        resp.raise_for_status()
        size = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > _MAX_PDF_BYTES:
                    raise ValueError(
                        f"PDF exceeds Slack file cap ({_MAX_PDF_BYTES // (1024*1024)} MB)"
                    )
                fh.write(chunk)
    log.info("Downloaded %s -> %s (%d bytes)", file_url_private, dest, size)


def _make_progress_callback(client: WebClient, job: JobRequest):
    """Closure that posts at 25/50/75% milestone crossings only.

    100% is posted by `_process_job` after upload completes — we intentionally
    do not emit it from here, because "extraction done" isn't the same as
    "buyer has the file in Slack."
    """
    last_milestone: list[int] = [0]

    def cb(phase: str, pct: float, message: str) -> None:
        # "cached" fires when an existing sidecar short-circuits the pipeline.
        if phase == "cached":
            _safe_post(client, job, formatting.cached_message(job.file_name))
            last_milestone[0] = 100
            return
        milestone = (int(pct * 100) // 25) * 25
        if milestone <= last_milestone[0]:
            return
        if milestone in (25, 50, 75):
            _safe_post(
                client, job,
                formatting.progress_message(milestone, phase, message),
            )
        last_milestone[0] = milestone

    return cb


_FIELDS_FOR_FILL_RATE = (
    "sku", "brand", "description", "color", "standard_color",
    "mg", "sg", "ssg", "intro_date", "usd_cost",
)


def _build_summary(extraction, write_stats: Optional[dict] = None) -> dict:
    s = oracle_summary(extraction)
    total = sum(st["total"] for st in s["per_field"].values())
    passing = sum(st["correct"] for st in s["per_field"].values())

    # Count amber / red cells across every card's per_field map. This mirrors
    # the rendering thresholds in write.py so the user sees exactly what they
    # got: amber = "review," red = "auto-blanked."
    amber = 0
    red = 0
    for c in extraction.confidence:
        for conf_value in c.per_field.values():
            if conf_value <= CONTRADICTED_THRESHOLD:
                red += 1
            elif conf_value < LOW_CONFIDENCE_THRESHOLD:
                amber += 1

    # Brand split: read directly off the extraction's layout metadata + cards.
    distinct_brands = {
        (card.brand or "").strip().upper()
        for card in extraction.all_cards
        if card.brand and card.brand.strip()
    }
    brand_count = len(distinct_brands) or 1
    is_multi_brand = brand_count > 1

    # Per-field FILL rate (populated vs total cards) — distinct from the oracle
    # verify rate, which only counts cards where the field had a value at all.
    # Missing fields are invisible in oracle_summary; this surfaces them.
    n_cards = len(extraction.all_cards)
    fill_rates: dict[str, float] = {}
    for f in _FIELDS_FOR_FILL_RATE:
        populated = sum(
            1 for card in extraction.all_cards
            if getattr(card, f, None) not in (None, "")
        )
        fill_rates[f] = (populated / n_cards) if n_cards else 0.0

    # Pages that errored during extraction — today these are silently swallowed
    # in the cards total. Surface so partial failures don't hide.
    pages_failed = [pe.page for pe in extraction.pages if pe.error]

    truncated_cards = (write_stats or {}).get("truncated_cards", 0)
    embedded_photos = (write_stats or {}).get("embedded_photos", 0)

    return {
        "cards": n_cards,
        "cells_pass": passing,
        "cells_total": total,
        "cells_amber": amber,
        "cells_red": red,
        "cost_usd": extraction.cost_usd,
        "is_multi_brand": is_multi_brand,
        "brand_count": brand_count,
        "fill_rates": fill_rates,
        "pages_failed": pages_failed,
        "truncated_cards": truncated_cards,
        "embedded_photos": embedded_photos,
    }


def _upload_file(client: WebClient, job: JobRequest, path: Path) -> None:
    try:
        client.files_upload_v2(
            channel=job.channel,
            thread_ts=job.thread_ts,
            file=str(path),
            filename=path.name,
            title=path.name,
            initial_comment=None,  # already posted done_message
        )
    except SlackApiError as exc:
        log.error("files_upload_v2 failed: %s", exc.response.data)
        _safe_post(
            client, job,
            formatting.error_message(
                job.file_name, "SlackUploadError",
                f"Workbook built but Slack rejected the upload: "
                f"{exc.response.data.get('error', 'unknown')}",
            ),
        )


def _safe_post(client: WebClient, job: JobRequest, text: str) -> None:
    """Post a thread reply; swallow Slack errors so they don't kill the worker."""
    try:
        client.chat_postMessage(
            channel=job.channel,
            thread_ts=job.thread_ts,
            text=text,
        )
    except SlackApiError as exc:
        log.error("chat_postMessage failed for %s: %s",
                  job.file_name, exc.response.data)


# Module-level singleton wired up by app.py at startup. Keeping it global means
# the Bolt event handler stays a thin shim.
_QUEUE: Optional[JobQueue] = None


def init_queue(client: WebClient, bot_token: str) -> JobQueue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = JobQueue(client, bot_token)
    return _QUEUE


def get_queue() -> JobQueue:
    if _QUEUE is None:
        raise RuntimeError("JobQueue not initialised — call init_queue() first")
    return _QUEUE


# Allow override in tests
def reset_queue_for_testing() -> None:
    global _QUEUE
    _QUEUE = None


# Pull the bot token out of the env for the downloader. We keep this lazy so
# importing this module never crashes when SLACK_BOT_TOKEN isn't set (e.g.
# during unit tests that don't exercise the worker).
def env_bot_token() -> str:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN env var is required")
    return token
