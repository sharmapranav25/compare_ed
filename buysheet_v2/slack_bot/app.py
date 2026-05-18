"""Slack Bolt app — file-upload event handler.

Listens for PDFs dropped in the configured channel and hands each one off to
the JobQueue worker. Replies in the upload's thread so progress + final
artefact stay attached to the original message.

Run via `python -m buysheet_v2.slack_bot`.
"""
from __future__ import annotations

import logging
import os

from slack_bolt import App
from slack_sdk import WebClient

from buysheet_v2.slack_bot import formatting, jobs

log = logging.getLogger(__name__)


def _allowed_channel(channel_id: str) -> bool:
    """Scope the bot to one channel. Empty SLACK_CHANNEL_ID = accept anywhere."""
    allowed = os.environ.get("SLACK_CHANNEL_ID", "").strip()
    return not allowed or channel_id == allowed


def build_app() -> App:
    """Wire up the Bolt app + event handlers.

    Build-time, not import-time, so missing tokens surface in `main()` rather
    than as an import error when tests touch this module.
    """
    bot_token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError(
            "SLACK_BOT_TOKEN env var is required. "
            "Get one from api.slack.com/apps -> OAuth & Permissions.",
        )

    app = App(token=bot_token)

    # Initialise the queue with a dedicated WebClient so the worker thread
    # doesn't share Bolt's internal client state.
    job_client = WebClient(token=bot_token)
    jobs.init_queue(job_client, bot_token)

    @app.event("message")
    def on_message(event, logger):
        # We only care about file-upload messages. Slack sends many other
        # subtypes (channel_join, etc) — ignore them silently.
        if event.get("subtype") != "file_share":
            return

        channel = event.get("channel")
        if not _allowed_channel(channel):
            return

        thread_ts = event.get("thread_ts") or event.get("ts")
        user_id = event.get("user", "unknown")
        files = event.get("files", []) or []

        if not files:
            return

        queue_ = jobs.get_queue()
        for f in files:
            file_id = f.get("id")
            file_name = f.get("name", "uploaded.pdf")
            file_type = (f.get("filetype") or "").lower()
            url_private = f.get("url_private")

            if file_type != "pdf":
                _post_skip(
                    job_client, channel, thread_ts,
                    formatting.skip_message(
                        file_name,
                        f"only PDFs are supported (got `.{file_type or 'unknown'}`)",
                    ),
                )
                continue
            if not url_private:
                _post_skip(
                    job_client, channel, thread_ts,
                    formatting.skip_message(
                        file_name, "Slack didn't return a download URL",
                    ),
                )
                continue

            request = jobs.JobRequest(
                file_id=file_id,
                file_name=file_name,
                file_url_private=url_private,
                channel=channel,
                thread_ts=thread_ts,
                user_id=user_id,
            )
            position = queue_.enqueue(request)
            if position > 0:
                # Already-busy ack; the real ack (with ETA) comes from the
                # worker once it picks up the job.
                _post_skip(
                    job_client, channel, thread_ts,
                    formatting.ack_message(
                        file_name, page_count=0, queue_position=position,
                    ),
                )

    # Catch-all for the file_shared event — Slack sometimes delivers it
    # alongside message.file_share, sometimes solo (depending on workspace
    # config). Acking here would double-fire, so we explicitly ignore it.
    @app.event("file_shared")
    def on_file_shared(event, logger):
        return

    return app


def _post_skip(client: WebClient, channel: str, thread_ts: str, text: str) -> None:
    try:
        client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text)
    except Exception as exc:
        log.error("Skip-message post failed: %s", exc)
