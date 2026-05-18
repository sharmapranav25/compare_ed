"""Slack bot for the buy-sheet pipeline.

User UX:
  1. Drops PDF in the configured channel
  2. Receives an ack with estimated time
  3. Pinged at 25 / 50 / 75% milestones
  4. Pinged at 100% with the finished xlsx attached + summary

Entry: `python -m buysheet_v2.slack_bot`
"""
