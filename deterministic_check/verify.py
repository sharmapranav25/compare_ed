"""Deterministic text-layer verification of VLM-extracted product fields.

Sits between Stage 1 (Opus visual extract) and Stage 2 (Sonnet SKU
validator) in extract_products.process_page. For pages whose PDF text
layer is readable, every field the VLM emitted is checked for physical
presence in that text layer using a canonicalized substring match.

Canonicalization: strip whitespace, uppercase. So "A24021 C" → "A24021C"
and the page text "...a24021c..." → "...A24021C...". Either side is
contained in the other if and only if the printed string matches.

Per-field policy (see plan along-side-this-i-valiant-ocean.md):
  sku         → row dropped if not in text layer
  cost/retail → value cleared (None), verification flag set
  description/color/intro_date → value kept, verification flag set
  gender_hint → skipped (comes from prev_context.current_section, not
                from the page)

Failure handling (text_layer_present=False, route through Stage 2 LLM):
  1. pdf[page_no-1].get_text() raises  → text_layer_error set with cause
  2. canonicalized text == ""           → scanned / image-only page
  3. text is OCR garbage                → handled implicitly (every
     candidate is dropped as not_in_text_layer, page ends up with 0
     products, existing REVIEW predicate flags it)
"""

from __future__ import annotations

import re
import threading
from typing import Any

import fitz

_FIELDS_DROP_ROW = ("sku",)
_FIELDS_CLEAR = ("cost", "retail")
_FIELDS_KEEP = ("description", "color", "intro_date")
# gender_hint is intentionally excluded — it's sourced from
# prev_context.current_section, not from the page itself.


def canon(s: str | None) -> str:
    """Strip all whitespace, uppercase. Returns "" for None / empty input."""
    return re.sub(r"\s+", "", s or "").upper()


def verify_against_text_layer(
    candidates: list[dict],
    pdf: fitz.Document,
    page_no: int,
    pdf_lock: threading.Lock,
) -> tuple[list[dict], list[dict], bool, str | None]:
    """Verify VLM-extracted candidates against the PDF text layer.

    Returns (kept, dropped, text_layer_present, text_layer_error).

    On any text-layer failure (raise OR empty), returns the candidates
    list verbatim with text_layer_present=False so the caller can route
    through today's Stage 2 LLM validator unchanged.
    """
    try:
        with pdf_lock:
            text = pdf[page_no - 1].get_text()
    except Exception as exc:  # noqa: BLE001
        return (list(candidates), [], False,
                f"pymupdf_failed: {type(exc).__name__}: {exc}")

    text_canon = canon(text)
    if not text_canon:
        return list(candidates), [], False, None

    kept: list[dict] = []
    dropped: list[dict] = []
    for c in candidates:
        sku_raw = c.get("sku")
        if not sku_raw or canon(str(sku_raw)) not in text_canon:
            entry: dict[str, Any] = dict(c)
            entry["reason"] = "sku_not_in_text_layer"
            entry["stage"] = "deterministic"
            dropped.append(entry)
            continue

        verification: dict[str, str] = {"sku": "ok"}
        out = dict(c)

        for field in _FIELDS_CLEAR:
            v = c.get(field)
            if v and canon(str(v)) in text_canon:
                verification[field] = "ok"
            elif v:
                verification[field] = "not_in_text_layer"
                out[field] = None
            else:
                verification[field] = "ok"  # nothing to verify

        for field in _FIELDS_KEEP:
            v = c.get(field)
            if v and canon(str(v)) in text_canon:
                verification[field] = "ok"
            elif v:
                verification[field] = "unverified"
            else:
                verification[field] = "ok"  # nothing to verify

        out["verification"] = verification
        kept.append(out)

    return kept, dropped, True, None
