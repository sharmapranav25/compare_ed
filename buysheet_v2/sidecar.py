"""Per-page sidecar persistence for resumability and partial-progress recovery.

KithxKeelo's default pipeline writes a single `<pdf_stem>.v2.cards.json`
covering every page — convenient but coarse. A crash mid-extract leaves no
recoverable state; any prompt tweak forces a full re-run.

This module adds per-page sidecar files at `<pdf_stem>.v2.pages/NN.json`
written incrementally inside the extract loop. Each file is a serialized
`PageExtraction` (matching the existing schema). Properties:

  - Crash recovery: extract that fails on page 47 still leaves pages 1-46
    on disk; a rerun can skip those (or future C6 backfill CLI can re-run
    just one stage on cached pages).
  - Tools (enrich_description_map, judge_existing, eval_against_goldens)
    keep reading the consolidated `.v2.cards.json` unchanged — that file
    is still written at the end of `run_pipeline` for back-compat. The
    per-page files are an *additive* write.
  - Disk format mirrors parent repo's `<doc>.pages/NN.json` so muscle
    memory carries over.

The resume path in pipeline.py (top of run_pipeline) is unchanged: it
still loads from the consolidated sidecar. C5's minimal scope is just to
make the per-page files exist; a future commit can migrate resume to
prefer per-page when present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from buysheet_v2.schemas.extraction_result import PageExtraction


def sidecar_dir(pdf_path: Path) -> Path:
    """Per-PDF directory holding one JSON per page."""
    return pdf_path.with_suffix("").with_name(f"{pdf_path.stem}.v2.pages")


def page_sidecar_path(pdf_path: Path, page_no: int) -> Path:
    """Absolute path to one page's sidecar file. Zero-padded so directory
    listing sorts in page order."""
    return sidecar_dir(pdf_path) / f"{page_no:02d}.json"


def save_page_sidecar(
    pdf_path: Path, page_no: int, page_extraction: PageExtraction,
) -> Path:
    """Persist one PageExtraction to disk. Returns the written path.

    Idempotent — overwrites prior content. Caller is responsible for not
    calling this concurrently with itself for the same (pdf_path, page_no)
    (per-page parallelism is not the concern here; pipeline.py's loop is
    serial across pages).
    """
    p = page_sidecar_path(pdf_path, page_no)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page_extraction.model_dump_json(indent=2, exclude_none=False))
    return p


def load_page_sidecar(pdf_path: Path, page_no: int) -> Optional[PageExtraction]:
    """Read one PageExtraction from disk. Returns None if absent or unparseable.

    Corruption-tolerant: a malformed sidecar (truncated JSON from a crash
    mid-write) is treated as missing — the caller will re-extract that
    page.
    """
    p = page_sidecar_path(pdf_path, page_no)
    if not p.exists():
        return None
    try:
        return PageExtraction.model_validate_json(p.read_text())
    except Exception:  # noqa: BLE001 — JSON, validation, or IO; any is "stale, redo"
        return None


def list_cached_pages(pdf_path: Path) -> list[int]:
    """Return the sorted list of page numbers that have a sidecar on disk."""
    d = sidecar_dir(pdf_path)
    if not d.exists():
        return []
    pages: list[int] = []
    for p in d.glob("*.json"):
        try:
            pages.append(int(p.stem))
        except ValueError:
            continue
    return sorted(pages)
