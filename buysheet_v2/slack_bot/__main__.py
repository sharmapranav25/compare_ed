"""Entry point: `python -m buysheet_v2.slack_bot`.

Loads .env, builds the Bolt app, opens a Socket Mode connection. No public
URL is required — Socket Mode opens a websocket from this process out to
Slack, which means the bot can run from a laptop, a Mac mini, or any
always-on box without exposing a port.
"""
from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode import SocketModeHandler

from buysheet_v2.slack_bot.app import build_app


def main() -> int:
    # .env in the repo root — same file the rest of the CLI uses.
    load_dotenv()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("buysheet_v2.slack_bot")

    app_token = os.environ.get("SLACK_APP_TOKEN", "").strip()
    if not app_token:
        log.error(
            "SLACK_APP_TOKEN env var is required for Socket Mode. "
            "Get one from api.slack.com/apps -> Basic Information -> "
            "App-Level Tokens (with connections:write scope).",
        )
        return 2

    try:
        app = build_app()
    except RuntimeError as exc:
        log.error("Bot startup failed: %s", exc)
        return 2

    channel_id = os.environ.get("SLACK_CHANNEL_ID", "").strip()
    if channel_id:
        log.info("Bot scoped to channel %s — uploads elsewhere will be ignored.",
                 channel_id)
    else:
        log.warning(
            "SLACK_CHANNEL_ID not set — bot will accept PDFs from any channel "
            "it's been invited to. Set SLACK_CHANNEL_ID in .env to scope.",
        )

    handler = SocketModeHandler(app, app_token)
    log.info("Connecting to Slack via Socket Mode...")
    handler.start()  # blocks until SIGINT / SIGTERM
    return 0


if __name__ == "__main__":
    sys.exit(main())
