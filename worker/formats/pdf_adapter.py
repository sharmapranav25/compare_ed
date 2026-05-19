"""PDF adapter — pure delegation to the existing single-doc pipeline.

Runs classify_pdf + extract_pdf against a PDF and returns the path of its
.pages/ directory. No logic duplicated; the existing pipeline owns
behavior, this module is just a uniform-shape entry point so run_multi.py
can dispatch by file type without knowing about PDF internals.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from classify_pages import classify_pdf  # noqa: E402
from extract_products import MODEL_OPUS, MODEL_SONNET, extract_pdf  # noqa: E402


def normalize(pdf_path: Path, *, workers: int, force: bool,
              model: str = "opus") -> Path:
    """Classify + extract a PDF. Returns the resulting .pages/ directory."""
    classify_pdf(pdf_path, only_page=None, force=force, workers=workers)
    mdl = MODEL_OPUS if model == "opus" else MODEL_SONNET
    extract_pdf(pdf_path, only_page=None, force=force, workers=workers, model=mdl)
    return pdf_path.with_name(pdf_path.stem + ".pages")
