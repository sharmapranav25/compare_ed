"""Merge per-doc .pages outputs into one products_rows list.

Each doc must already be normalized (its .pages/*.json files written by
the relevant adapter). Reads all of them in CLI/priority order, groups
products by canonicalized SKU, and merges field-by-field with
first-non-empty-wins per priority order.

Conflict policy:
  When the same canonical SKU appears in multiple sources/pages AND any
  non-empty field disagrees across them (after appropriate value
  normalization), the rows are KEPT SEPARATE rather than merged. Both
  rows are flagged (yellow STYLE# tint + entry in SKU_CONFLICTS sheet)
  so a reviewer can see and resolve the conflict instead of the merge
  silently picking one value. Identical-value duplicates still collapse.

Returns the same products_rows + page_records shape build_buysheet's
write helpers consume — so build_merged.py can hand the result straight
to gather_unique_tasks / write_rows without any field translation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_buysheet import parse_money  # noqa: E402

# Order matters for the merge precedence inside a SKU group. "sku" is
# included so that the first non-empty SKU printing (with whitespace /
# punctuation preserved) wins for display when the group collapses.
# image_path follows first-non-empty-wins: when only one source carries
# a crop (PDF with detection enabled), it survives untouched; when both
# do it's a single-doc-only feature so the case shouldn't arise.
_MERGE_FIELDS = ("sku", "description", "color", "cost",
                 "retail", "intro_date", "gender_hint", "image_path")

# Fields where equivalence is numeric, not string.
_NUMERIC_FIELDS = ("cost", "retail")


def _canon_sku(s: str | None) -> str:
    return re.sub(r"\s+", "", (s or "")).upper()


def _has_value(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    return True


def _norm_text(v) -> str:
    """Collapse whitespace, strip, lowercase. Used for non-numeric
    equivalence so "HANDBALL SPEZIAL" == "handball  spezial"."""
    return re.sub(r"\s+", " ", str(v).strip()).lower()


def _values_equivalent(field: str, a, b) -> bool:
    """Field-aware equivalence. cost/retail compared as parsed numbers
    (so 'WHSL $120' == 'MSRP 120 USD' == 120.0); text fields compared
    after whitespace+case normalization."""
    if field in _NUMERIC_FIELDS:
        na, _ = parse_money(a)
        nb, _ = parse_money(b)
        if na is not None and nb is not None:
            return round(na, 2) == round(nb, 2)
        # If either fails to parse, fall back to text comparison.
    return _norm_text(a) == _norm_text(b)


def _conflicts_in_group(group: list[dict]) -> dict[str, list]:
    """For each field, return the distinct non-empty values seen across
    the group (in first-seen order). A field appears in the result only
    if there are ≥2 mutually-non-equivalent values — i.e., a real
    conflict, not just one source having a blank."""
    out: dict[str, list] = {}
    for field in _MERGE_FIELDS:
        if field == "sku":
            # By construction every entry in the group has the same canon
            # SKU, so the raw spellings differ only in whitespace/case —
            # not a conflict worth flagging.
            continue
        if field == "image_path":
            # Derived crop path, not vendor data. Two different files
            # mean two different docs, not a vendor disagreement.
            continue
        distinct: list = []
        for entry in group:
            v = entry["product"].get(field)
            if not _has_value(v):
                continue
            if any(_values_equivalent(field, v, existing) for existing in distinct):
                continue
            distinct.append(v)
        if len(distinct) > 1:
            out[field] = distinct
    return out


def _page_is_flagged(rec: dict, products: list[dict]) -> bool:
    """Same flagging predicate build_buysheet.collect_pages uses, kept in
    sync deliberately so the merged REVIEW sheet matches expectations."""
    if rec.get("error"):
        return True
    if rec.get("label") == "product" and not products:
        return True
    if rec.get("text_layer_present") is False:
        return True
    for p in products:
        for marker in (p.get("verification") or {}).values():
            if marker not in ("ok", "deterministic"):
                return True
    return False


def _read_doc(pages_dir: Path) -> tuple[list[dict], list[dict], str | None]:
    """Walk one doc's .pages/*.json in order. Returns:
       (per-product entries with page_no/context, page_records, vendor)."""
    entries: list[dict] = []
    page_records: list[dict] = []
    vendor: str | None = None
    for jf in sorted(pages_dir.glob("*.json")):
        if jf.name.startswith("_"):
            continue
        rec = json.loads(jf.read_text())
        ctx = rec.get("context_after") or rec.get("prev_context") or {}
        if ctx.get("vendor") and not vendor:
            vendor = ctx["vendor"]
        ps = rec.get("products") or []
        flagged = _page_is_flagged(rec, ps)
        page_no = rec.get("page_no")
        for p in ps:
            entries.append({
                "page_no": page_no,
                "context": ctx,
                "product": p,
                "page_flagged": flagged,
            })
        page_records.append({
            "page_no": page_no,
            "label": rec.get("label", "unknown"),
            "error": rec.get("error"),
            "n_products": len(ps),
            "flagged": flagged,
        })
    return entries, page_records, vendor


def _row_from_entry(entry: dict, page_flagged_override: bool | None = None) -> dict:
    """Build a products_rows entry from a single source entry. Used for
    the conflict-kept-separate path."""
    return {
        "page_no": entry["page_no"],
        "context": entry["context"],
        "product": entry["product"],
        "page_flagged": (page_flagged_override
                         if page_flagged_override is not None
                         else entry["page_flagged"]),
    }


def _collapse_group(group: list[dict]) -> dict:
    """Merge a non-conflicting group into one products_rows entry using
    the existing first-non-empty-wins-per-field policy."""
    merged_product: dict = {}
    field_winner: dict[str, dict] = {}
    for field in _MERGE_FIELDS:
        for entry in group:
            v = entry["product"].get(field)
            if _has_value(v):
                merged_product[field] = v
                field_winner[field] = entry
                break
    verification: dict = {}
    for field, entry in field_winner.items():
        marker = (entry["product"].get("verification") or {}).get(field)
        if marker is not None:
            verification[field] = marker
    if verification:
        merged_product["verification"] = verification
    winning_entry = group[0]
    return {
        "page_no": winning_entry["page_no"],
        "context": winning_entry["context"],
        "product": merged_product,
        "page_flagged": any(e["page_flagged"] for e in group),
    }


def merge_docs(doc_paths: list[Path], pages_dirs: list[Path]
               ) -> tuple[list[dict], list[dict], list[str | None], list[dict]]:
    """Merge across N normalized docs.

    Args:
      doc_paths : the original input paths, used for source attribution
      pages_dirs: the corresponding .pages directories (same order)

    Returns:
      products_rows  : [{page_no, context, product, page_flagged}, ...]
                       ready for build_buysheet.write_rows /
                       gather_unique_tasks
      page_records   : union of per-doc page records with `source`
                       filename attached; consumed by build_merged for
                       the REVIEW sheet
      vendors_per_doc: one entry per input doc (None if no vendor
                       signal), so the caller can enforce single-vendor
                       policy
      sku_conflicts  : [{sku_canon, conflicting_fields, occurrences}]
                       — one entry per SKU group where any field
                       disagreed. Consumed by build_merged to populate
                       the SKU_CONFLICTS sheet.
    """
    if len(doc_paths) != len(pages_dirs):
        raise ValueError("doc_paths and pages_dirs must be same length")

    buckets: dict[str, list[dict]] = {}
    order: list[str] = []
    merged_page_records: list[dict] = []
    vendors_per_doc: list[str | None] = []

    for pri, (doc_path, pages_dir) in enumerate(zip(doc_paths, pages_dirs)):
        entries, page_records, doc_vendor = _read_doc(pages_dir)
        vendors_per_doc.append(doc_vendor)
        for e in entries:
            sku_canon = _canon_sku(e["product"].get("sku"))
            if not sku_canon:
                continue
            e["priority"] = pri
            e["source"] = doc_path.name
            if sku_canon not in buckets:
                buckets[sku_canon] = []
                order.append(sku_canon)
            buckets[sku_canon].append(e)
        for rec in page_records:
            merged_page_records.append({**rec, "source": doc_path.name})

    products_rows: list[dict] = []
    sku_conflicts: list[dict] = []

    for sku_canon in order:
        group = sorted(buckets[sku_canon], key=lambda e: e["priority"])
        conflicts = _conflicts_in_group(group)

        if not conflicts:
            products_rows.append(_collapse_group(group))
            continue

        # Conflict — keep every occurrence as its own row, force-flagged
        # so the STYLE# cell gets the yellow REVIEW tint in the xlsx.
        for entry in group:
            products_rows.append(_row_from_entry(entry, page_flagged_override=True))
        sku_conflicts.append({
            "sku_canon": sku_canon,
            "conflicting_fields": sorted(conflicts.keys()),
            "occurrences": [
                {
                    "source": entry["source"],
                    "page_no": entry["page_no"],
                    "sku_raw": entry["product"].get("sku"),
                    "values": {f: entry["product"].get(f) for f in _MERGE_FIELDS
                               if f in conflicts},
                }
                for entry in group
            ],
        })

    return products_rows, merged_page_records, vendors_per_doc, sku_conflicts
