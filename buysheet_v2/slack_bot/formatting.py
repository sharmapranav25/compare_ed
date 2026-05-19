"""Slack message templates.

Plain-text strings only — no Block Kit, so what Claude renders here is exactly
what the buyer sees. Centralised so all copy lives in one file and is easy to
tweak without touching event handlers or job logic.
"""
from __future__ import annotations

from typing import Optional

# Calibration: Nike 9pp = ~3 min, Adidas 31pp = ~10 min from README.
# 20 s/page is the central estimate; we add a 25% buffer to bias the
# user-facing ETA toward "early" rather than "late."
SECONDS_PER_PAGE_ESTIMATE = 25


def format_eta(page_count: int) -> str:
    total_sec = page_count * SECONDS_PER_PAGE_ESTIMATE
    if total_sec < 90:
        return f"~{total_sec}s"
    minutes = round(total_sec / 60)
    return f"~{minutes} min"


def ack_message(file_name: str, page_count: int, queue_position: int = 0) -> str:
    eta = format_eta(page_count)
    if queue_position > 0:
        return (
            f":page_facing_up: Got *{file_name}* ({page_count} pages). "
            f"Queued behind {queue_position} other job"
            f"{'s' if queue_position > 1 else ''}. "
            f"I'll start as soon as the current one finishes."
        )
    return (
        f":page_facing_up: Got *{file_name}* ({page_count} pages). "
        f"Analyzing the catalog and building your buy sheet now.\n"
        f"Estimated time: *{eta}*. I'll ping you at 25 / 50 / 75 / 100%."
    )


def progress_message(milestone_pct: int, phase: str, info: str) -> str:
    label = {25: "Quarter-way there", 50: "Halfway", 75: "Almost done"}.get(
        milestone_pct, f"{milestone_pct}%",
    )
    return f"*{milestone_pct}% — {label}.* {info}"


def cached_message(file_name: str) -> str:
    return (
        f":zap: I've already extracted *{file_name}* in a previous run — "
        f"sending the cached buy sheet now (no re-extraction cost)."
    )


def _format_fill_rates(fill_rates: dict) -> str:
    """Compact one-line per-field fill rates, with ⚠ on low fields."""
    parts = []
    for f, rate in fill_rates.items():
        pct = int(round(rate * 100))
        marker = " ⚠" if rate < 0.5 else ""
        # Trim long field names so the line stays scannable in Slack
        short = {"description": "desc", "standard_color": "std_col",
                 "intro_date": "intro", "usd_cost": "cost"}.get(f, f)
        parts.append(f"{short} {pct}%{marker}")
    return " · ".join(parts)


def done_message(
    file_name: str,
    *,
    cards: int,
    cells_pass: int,
    cells_total: int,
    cells_amber: int,
    cells_red: int,
    cost_usd: float,
    is_multi_brand: bool,
    brand_count: int,
    fill_rates: Optional[dict] = None,
    pages_failed: Optional[list] = None,
    truncated_cards: int = 0,
    embedded_photos: int = 0,
) -> str:
    pct = (100 * cells_pass / cells_total) if cells_total else 0.0
    brand_blurb = (
        f"Split into *{brand_count}* tabs by brand."
        if is_multi_brand
        else "Single-brand catalog — one tab."
    )
    extracted_line = f"• *{cards}* product cards extracted"
    if pages_failed:
        n = len(pages_failed)
        # Only list page numbers if there aren't too many; otherwise just the count
        if n <= 8:
            extracted_line += f"  (⚠ {n} page{'s' if n != 1 else ''} failed: {', '.join(f'p{p}' for p in pages_failed)})"
        else:
            extracted_line += f"  (⚠ {n} pages failed during extraction)"

    parts = [
        f":white_check_mark: *Done* — buy sheet for *{file_name}* attached.",
        "",
        extracted_line,
        f"• *{pct:.1f}%* of cells verified against source text "
        f"({cells_pass}/{cells_total})",
    ]
    if cells_amber:
        parts.append(
            f"• :large_yellow_square: *{cells_amber}* cells amber-flagged "
            f"(model is uncertain — hover for source + accept/edit)"
        )
    if cells_red:
        parts.append(
            f"• :red_square: *{cells_red}* cells left blank "
            f"(the model's value contradicted the source — manual review required)"
        )
    if truncated_cards:
        parts.append(
            f"• :warning: *{truncated_cards}* cards dropped — workbook hard-cap at row 883. "
            f"Consider splitting the catalog and re-running."
        )
    if embedded_photos:
        parts.append(f"• *{embedded_photos}* product photos embedded in col A")
    if fill_rates:
        parts.append("• Fill rate per field:")
        parts.append(f"  `{_format_fill_rates(fill_rates)}`")
    parts.append(f"• {brand_blurb}")
    parts.append(f"• Cost: *${cost_usd:.2f}*")
    parts.append("")
    parts.append(
        "_Hover any coloured cell in Excel to see which page + what the model "
        "extracted. Open the source PDF for the cited page to verify._"
    )
    return "\n".join(parts)


def error_message(file_name: str, exc_type: str, summary: str,
                  pages_done: Optional[int] = None) -> str:
    if pages_done:
        return (
            f":warning: Hit a problem partway through *{file_name}* "
            f"(failed after {pages_done} page{'s' if pages_done != 1 else ''}).\n"
            f"`{exc_type}: {summary}`\n"
            f"I won't auto-retry — share the PDF again if you'd like me to try a fresh run."
        )
    return (
        f":x: Couldn't process *{file_name}*.\n"
        f"`{exc_type}: {summary}`"
    )


def skip_message(file_name: str, reason: str) -> str:
    return f"Skipping *{file_name}* — {reason}."
