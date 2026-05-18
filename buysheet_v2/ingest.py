"""PDF ingestion: render every page to PNG + extract text layer.

Uses both pypdfium2 (for rendering — better PNG quality + crop support) and
PyMuPDF (for text + text-search SKU anchoring). Results are cached in the
PDF's parent directory so re-runs after prompt tweaks are free.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import pypdfium2 as pdfium

from buysheet_v2.lifted.pdf_render import VLM_MAX_LONG_EDGE_PX, render_page_for_vlm


@dataclass
class IngestedPage:
    """One page's renderable + searchable state.

    png_bytes is sized for VLM consumption (max long-edge VLM_MAX_LONG_EDGE_PX),
    so width_px / height_px are the EXACT coordinate system the VLM operates
    in. Card bboxes returned by cards.py are in this space.
    """

    page_no: int  # 1-indexed
    png_bytes: bytes  # VLM-sized render, deterministic coord system
    text: str  # full page text via PyMuPDF
    width_px: int  # actual PNG width
    height_px: int  # actual PNG height


@dataclass
class IngestedPDF:
    """All pages of a PDF, ready for VLM processing."""

    pdf_path: Path
    pages: list[IngestedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)


def ingest(pdf_path: Path, *, max_long_edge: int = VLM_MAX_LONG_EDGE_PX) -> IngestedPDF:
    """Render every page at VLM-friendly size + extract per-page text."""
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pdfium_doc = pdfium.PdfDocument(str(pdf_path))
    fitz_doc = fitz.open(str(pdf_path))
    pages: list[IngestedPage] = []
    try:
        for i in range(len(pdfium_doc)):
            page_no = i + 1
            png, w, h = render_page_for_vlm(pdfium_doc, page_no, max_long_edge=max_long_edge)
            text = fitz_doc.load_page(i).get_text("text") or ""
            pages.append(IngestedPage(
                page_no=page_no, png_bytes=png, text=text,
                width_px=w, height_px=h,
            ))
    finally:
        fitz_doc.close()

    return IngestedPDF(pdf_path=pdf_path, pages=pages)


# --- SKU anchoring -----------------------------------------------------------

# Loose SKU regex — covers most shoebuyer formats:
#   Nike: IX9999-999, JA1066-100
#   Adidas: KH9999, KJ7413
#   Hanger: M044-SHOE, MBU032M-BOOT
#   Converse: A24020C
# We deliberately allow noise here; the VLM filters by visual context.
SKU_REGEX_CANDIDATES = [
    r"\b[A-Z]{1,4}\d{3,6}[A-Z]?(?:-[A-Z0-9]+)?\b",   # most common
    r"\b\d{3,6}[A-Z]+\b",                              # digit-prefix variants
    r"\bSKU\s+([A-Z0-9][A-Z0-9\-]+)\b",                # explicit "SKU XXX" form
]


def anchor_skus_on_page(text: str) -> list[str]:
    """Return distinct SKU-like tokens found in the page text.

    Vendor-agnostic regex pass. Used as HINTS for the VLM, not as ground truth.
    The VLM is the source of truth for SKU identity.
    """
    found: list[str] = []
    seen: set[str] = set()
    for pat in SKU_REGEX_CANDIDATES:
        for m in re.finditer(pat, text):
            tok = m.group(1) if m.groups() else m.group(0)
            tok = tok.strip()
            if len(tok) < 3 or len(tok) > 30:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            found.append(tok)
    return found


def find_sku_in_text(text: str, sku: str) -> tuple[int, int] | None:
    """Whitespace-tolerant search for an SKU in page text. Returns (start, end) offsets."""
    pat = re.compile(r"\s*".join(re.escape(c) for c in sku))
    m = pat.search(text)
    return (m.start(), m.end()) if m else None
