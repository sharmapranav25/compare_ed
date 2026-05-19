"""Three-way per-cell comparison: Sonnet vs Opus vs Gemini.

Takes 3 sidecars (one per model) from multi_model_extract and produces a
consolidated inconsistency report identifying every (SKU, field) where any
of the models disagree.

Per-cell agreement tiers:
  ALL_3_AGREE     all three models extracted the same normalized value
  2_OF_3_AGREE    two models agree, one dissents — usually the dissenter is wrong
  ALL_3_DISAGREE  three different values — worst case, no signal at all
  SONNET_ONLY     only Sonnet found a value for this (SKU, field)
  OPUS_ONLY       only Opus
  GEMINI_ONLY     only Gemini
  TWO_NULL        only one model extracted, two left null (asymmetric)

Output:
  - aggregate stats (% all-3-agree per field, per vendor)
  - top inconsistencies (which fields disagree most often)
  - per-SKU inconsistency table (which SKUs have the most cross-model conflicts)
  - markdown report saved to ~/buysheet_runs/MULTI_MODEL_COMPARE_<vendor>.md

Usage:
    python -m buysheet_v2.tools.three_way_compare <pdf_stem>.sonnet.v2.cards.json \\
                                                   <pdf_stem>.opus.v2.cards.json \\
                                                   <pdf_stem>.gemini.v2.cards.json \\
                                                   --vendor-key <key>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from buysheet_v2.schemas.extraction_result import CatalogExtraction

COMPARE_FIELDS = (
    "sku", "brand", "description", "color", "standard_color",
    "mg", "sg", "ssg", "intro_date", "usd_cost",
)


def _normalize(value) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def index_cards(ext: CatalogExtraction):
    """Index cards by (page, normalized SKU) for cross-model alignment."""
    return {(c.page, _normalize(c.sku)): c for c in ext.all_cards}


def compare_three(sonnet_path: Path, opus_path: Path, gemini_path: Path,
                  vendor_key: str) -> dict:
    sonnet = CatalogExtraction.model_validate_json(sonnet_path.read_text())
    opus = CatalogExtraction.model_validate_json(opus_path.read_text())
    gemini = CatalogExtraction.model_validate_json(gemini_path.read_text())

    s_idx = index_cards(sonnet)
    o_idx = index_cards(opus)
    g_idx = index_cards(gemini)

    all_keys = set(s_idx) | set(o_idx) | set(g_idx)
    only_sonnet = set(s_idx) - set(o_idx) - set(g_idx)
    only_opus = set(o_idx) - set(s_idx) - set(g_idx)
    only_gemini = set(g_idx) - set(s_idx) - set(o_idx)
    all_three_have = set(s_idx) & set(o_idx) & set(g_idx)

    # Per-field agreement tiers
    per_field_tiers = {f: Counter() for f in COMPARE_FIELDS}
    # Per-SKU disagreement count (for "top conflict SKUs")
    per_sku_conflicts = Counter()
    # Sample disagreements for the report
    samples = defaultdict(list)

    for key in all_three_have:
        s_card = s_idx[key]
        o_card = o_idx[key]
        g_card = g_idx[key]
        for f in COMPARE_FIELDS:
            s_val = getattr(s_card, f, None)
            o_val = getattr(o_card, f, None)
            g_val = getattr(g_card, f, None)
            s_norm = _normalize(s_val)
            o_norm = _normalize(o_val)
            g_norm = _normalize(g_val)
            present = [
                ("sonnet", s_val, s_norm),
                ("opus",   o_val, o_norm),
                ("gemini", g_val, g_norm),
            ]
            non_null = [(name, raw, norm) for name, raw, norm in present if norm]
            if not non_null:
                per_field_tiers[f]["ALL_3_NULL"] += 1
                continue
            if len(non_null) == 1:
                tier = {
                    "sonnet": "SONNET_ONLY",
                    "opus":   "OPUS_ONLY",
                    "gemini": "GEMINI_ONLY",
                }[non_null[0][0]]
                per_field_tiers[f][tier] += 1
                continue
            if len(non_null) == 2:
                a, b = non_null[0], non_null[1]
                if a[2] == b[2]:
                    per_field_tiers[f]["TWO_AGREE_ONE_NULL"] += 1
                else:
                    per_field_tiers[f]["TWO_DISAGREE_ONE_NULL"] += 1
                    per_sku_conflicts[key] += 1
                    if len(samples[f]) < 12:
                        samples[f].append((s_card.sku, s_card.page,
                                          ("sonnet", s_val), ("opus", o_val), ("gemini", g_val)))
                continue
            # All 3 non-null
            norms = {a[2] for a in non_null}
            if len(norms) == 1:
                per_field_tiers[f]["ALL_3_AGREE"] += 1
            elif len(norms) == 2:
                per_field_tiers[f]["2_OF_3_AGREE"] += 1
                per_sku_conflicts[key] += 1
                if len(samples[f]) < 12:
                    samples[f].append((s_card.sku, s_card.page,
                                      ("sonnet", s_val), ("opus", o_val), ("gemini", g_val)))
            else:
                per_field_tiers[f]["ALL_3_DISAGREE"] += 1
                per_sku_conflicts[key] += 1
                if len(samples[f]) < 12:
                    samples[f].append((s_card.sku, s_card.page,
                                      ("sonnet", s_val), ("opus", o_val), ("gemini", g_val)))

    return {
        "vendor": vendor_key,
        "card_counts": {
            "sonnet": len(s_idx), "opus": len(o_idx), "gemini": len(g_idx),
            "all_three_have": len(all_three_have),
            "only_sonnet": len(only_sonnet),
            "only_opus": len(only_opus),
            "only_gemini": len(only_gemini),
        },
        "per_field_tiers": {f: dict(c) for f, c in per_field_tiers.items()},
        "per_sku_conflicts": per_sku_conflicts.most_common(10),
        "samples": dict(samples),
        "all_three_have": all_three_have,
    }


def format_report(result: dict) -> str:
    out = []
    def w(line=""): out.append(line)

    vendor = result["vendor"]
    w(f"# Multi-model comparison — {vendor}\n")
    w("Three independent extractions (Sonnet 4.6, Opus 4.7, Gemini 2.5 Flash) "
      "on the SAME card-bboxes from Sonnet's cards.py. Only the extract step "
      "differs across models.\n")

    cc = result["card_counts"]
    w("## Card extraction overview\n")
    w(f"- Sonnet extracted: **{cc['sonnet']}** cards")
    w(f"- Opus extracted:   **{cc['opus']}** cards")
    w(f"- Gemini extracted: **{cc['gemini']}** cards")
    w(f"- All 3 extracted same SKU: **{cc['all_three_have']}**")
    w(f"- Sonnet-only SKUs (Opus + Gemini both missed): {cc['only_sonnet']}")
    w(f"- Opus-only SKUs:   {cc['only_opus']}")
    w(f"- Gemini-only SKUs: {cc['only_gemini']}\n")

    w("## Per-field cross-model agreement\n")
    w("For SKUs all 3 models extracted, what fraction of cells had all 3 agree?")
    w("Higher = stronger cross-model consensus = more trustworthy field.\n")
    n_cards = max(1, cc["all_three_have"])
    w("| Field | All 3 agree | 2 of 3 agree | All 3 disagree | Any 1 only | 2 null/1 has | All null |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for f in COMPARE_FIELDS:
        c = result["per_field_tiers"][f]
        a3 = c.get("ALL_3_AGREE", 0)
        a2 = c.get("2_OF_3_AGREE", 0)
        ad = c.get("ALL_3_DISAGREE", 0)
        any_only = (c.get("SONNET_ONLY", 0) + c.get("OPUS_ONLY", 0)
                    + c.get("GEMINI_ONLY", 0))
        two_present = c.get("TWO_AGREE_ONE_NULL", 0) + c.get("TWO_DISAGREE_ONE_NULL", 0)
        null3 = c.get("ALL_3_NULL", 0)
        w(f"| **{f}** | {a3} ({100*a3/n_cards:.0f}%) | {a2} | {ad} | "
          f"{any_only} | {two_present} | {null3} |")
    w()

    if result["per_sku_conflicts"]:
        w("## Top SKUs with most cross-model conflicts\n")
        w("These SKUs have the most field-level disagreements among the 3 models — "
          "they're the highest-priority review candidates.\n")
        w("| SKU | Page | Conflict count |")
        w("|---|---:|---:|")
        for (page, sku_norm), n in result["per_sku_conflicts"]:
            w(f"| `{sku_norm}` | p{page} | {n} |")
        w()

    if result["samples"]:
        w("## Sample disagreements (per field, up to 5 each)\n")
        for f in COMPARE_FIELDS:
            ss = result["samples"].get(f, [])
            if not ss:
                continue
            w(f"### {f}\n")
            w("| SKU | Page | Sonnet | Opus | Gemini |")
            w("|---|---:|---|---|---|")
            for sku, page, s, o, g in ss[:5]:
                w(f"| `{sku}` | p{page} | {s[1]!r} | {o[1]!r} | {g[1]!r} |")
            w()
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(prog="three_way_compare")
    ap.add_argument("sonnet_sidecar", type=Path)
    ap.add_argument("opus_sidecar", type=Path)
    ap.add_argument("gemini_sidecar", type=Path)
    ap.add_argument("--vendor-key", required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output markdown path (default: ~/buysheet_runs/MULTI_MODEL_COMPARE_<vendor>.md)")
    args = ap.parse_args()

    for p in (args.sonnet_sidecar, args.opus_sidecar, args.gemini_sidecar):
        if not p.exists():
            print(f"sidecar not found: {p}", file=sys.stderr)
            return 1

    result = compare_three(args.sonnet_sidecar, args.opus_sidecar,
                           args.gemini_sidecar, args.vendor_key)
    report = format_report(result)

    out_path = args.out or (
        Path.home() / "buysheet_runs" / f"MULTI_MODEL_COMPARE_{args.vendor_key}.md"
    )
    out_path.write_text(report)
    print(report)
    print(f"\n\nReport saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
