"""Fill-rate analysis over a finished pipeline run.

Reads <pdf-stem>.pages/*.json (the output of classify_pages.py +
extract_products.py) and reports:

  - Page totals: total / classified `product` / text-layer-absent.
  - SKU coverage: kept rows / total Opus Stage-1 candidates.
  - Drop breakdown by stage (`deterministic` vs `llm_validator`).
  - Per-field fill rate (description / color / cost / retail / intro_date):
    of kept rows, fraction with a non-null/non-empty value.
  - Verification flag counts: how many `verification.<field>` markers
    are `"not_in_text_layer"` or `"unverified"` across kept rows.

Pure local compute — no LLM calls. Safe to re-run cheaply.

Usage:
  # Standalone (run after extract — does NOT require build to be done):
  python -m analysis.fill_rate files/<vendor>/<doc>.pdf

  # Auto-invoked by run_pipeline.py after build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analysis.usage import add_usage, cost_usd, empty_usage

ANALYSIS_DIR = Path(__file__).resolve().parent

_STEPS = ("classify", "extract", "validate", "vocab_map")

FIELDS = ("description", "color", "cost", "retail", "intro_date")
VERIFICATION_FLAGS = {
    "description": "unverified",
    "color":       "unverified",
    "cost":        "not_in_text_layer",
    "retail":      "not_in_text_layer",
    "intro_date":  "unverified",
}


def _is_filled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def compute_fill_rate(pdf_path: Path) -> dict:
    pages_dir = pdf_path.with_name(pdf_path.stem + ".pages")
    if not pages_dir.exists():
        raise FileNotFoundError(
            f"{pages_dir} not found — run classify_pages.py + "
            "extract_products.py first"
        )

    pages_total = 0
    pages_product = 0
    pages_text_layer_absent = 0
    candidates = 0
    kept = 0
    dropped_deterministic = 0
    dropped_llm_validator = 0
    field_filled = {f: 0 for f in FIELDS}
    flag_counts = {f"{f}_{flag}": 0 for f, flag in VERIFICATION_FLAGS.items()}
    usage_by_step: dict[str, dict] = {s: empty_usage() for s in _STEPS}

    for jf in sorted(pages_dir.glob("*.json")):
        # Skip the build's usage sidecar — it's read separately below.
        if jf.name.startswith("_"):
            continue
        rec = json.loads(jf.read_text())
        u = rec.get("usage") or {}
        for step in ("classify", "extract", "validate"):
            add_usage(usage_by_step[step], u.get(step) or {})
        pages_total += 1
        label = rec.get("label", "unknown")
        if label != "product":
            continue
        pages_product += 1
        if rec.get("text_layer_present") is False:
            pages_text_layer_absent += 1
        candidates += int(rec.get("n_candidates") or 0)
        products = rec.get("products") or []
        kept += len(products)
        for r in rec.get("rejected_candidates") or []:
            stage = r.get("stage")
            if stage == "deterministic":
                dropped_deterministic += 1
            elif stage == "llm_validator":
                dropped_llm_validator += 1
            else:
                # Legacy records without a stage field — count as LLM (the
                # pre-deterministic-check default).
                dropped_llm_validator += 1
        for p in products:
            for f in FIELDS:
                if _is_filled(p.get(f)):
                    field_filled[f] += 1
            verification = p.get("verification") or {}
            for f, expected in VERIFICATION_FLAGS.items():
                if verification.get(f) == expected:
                    flag_counts[f"{f}_{expected}"] += 1

    sku_coverage = (kept / candidates) if candidates else None
    field_fill_rate = {
        f: ((field_filled[f] / kept) if kept else None) for f in FIELDS
    }

    build_usage_path = pages_dir / "_build_usage.json"
    if build_usage_path.exists():
        build_usage = json.loads(build_usage_path.read_text())
        add_usage(usage_by_step["vocab_map"], build_usage.get("vocab_map") or {})

    cost_by_step = {s: round(cost_usd(usage_by_step[s]), 4) for s in _STEPS}
    cost_by_step["total"] = round(sum(cost_by_step.values()), 4)

    return {
        "pdf": pdf_path.stem,
        "totals": {
            "pages_total": pages_total,
            "pages_product": pages_product,
            "pages_text_layer_absent": pages_text_layer_absent,
            "candidates": candidates,
            "kept": kept,
            "dropped_deterministic": dropped_deterministic,
            "dropped_llm_validator": dropped_llm_validator,
        },
        "sku_coverage": sku_coverage,
        "field_fill_rate": field_fill_rate,
        "verification_flags": flag_counts,
        "usage": usage_by_step,
        "cost_usd": cost_by_step,
    }


def write_report(pdf_path: Path, report: dict) -> Path:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_DIR / f"{pdf_path.stem}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return out


def _pct(x) -> str:
    return f"{x * 100:5.1f}%" if isinstance(x, (int, float)) else "  n/a"


def print_summary(report: dict, report_path: Path | None = None) -> None:
    t = report["totals"]
    print("=== fill-rate ===", file=sys.stderr)
    print(f"  pages:        {t['pages_total']} total, "
          f"{t['pages_product']} product, "
          f"{t['pages_text_layer_absent']} without text layer",
          file=sys.stderr)
    sc = report["sku_coverage"]
    sc_str = f"{sc * 100:.1f}%" if isinstance(sc, (int, float)) else "n/a"
    print(f"  SKU coverage: {t['kept']} / {t['candidates']} candidates  "
          f"({sc_str})", file=sys.stderr)
    print(f"  dropped:      {t['dropped_deterministic']} deterministic, "
          f"{t['dropped_llm_validator']} llm_validator", file=sys.stderr)
    flags = report["verification_flags"]
    print(f"  field fill rate (of {t['kept']} kept rows):", file=sys.stderr)
    for f in FIELDS:
        rate = report["field_fill_rate"][f]
        flag_key = f"{f}_{VERIFICATION_FLAGS[f]}"
        n_flag = flags.get(flag_key, 0)
        suffix = (f"   ({n_flag} {VERIFICATION_FLAGS[f]})"
                  if n_flag else "")
        print(f"    {f:<12}{_pct(rate)}{suffix}", file=sys.stderr)
    costs = report.get("cost_usd") or {}
    usage = report.get("usage") or {}
    if costs:
        print(f"  LLM spend (USD, estimate):", file=sys.stderr)
        for s in _STEPS:
            u = usage.get(s) or {}
            n = int(u.get("n_calls") or 0)
            tok_in  = int(u.get("input_tokens") or 0)
            tok_out = int(u.get("output_tokens") or 0)
            tok_cr  = int(u.get("cache_read_input_tokens") or 0)
            tok_cw  = int(u.get("cache_creation_input_tokens") or 0)
            cache_str = ""
            if tok_cr or tok_cw:
                cache_str = f", cache r/w {tok_cr}/{tok_cw}"
            print(f"    {s:<10}${costs.get(s, 0.0):7.4f}  "
                  f"({n} calls, in/out {tok_in}/{tok_out}{cache_str})",
                  file=sys.stderr)
        print(f"    {'total':<10}${costs.get('total', 0.0):7.4f}",
              file=sys.stderr)
    if report_path is not None:
        print(f"  → {report_path}", file=sys.stderr)


def analyze(pdf_path: Path) -> Path:
    """One-call helper: compute + write + print. Returns report path."""
    report = compute_fill_rate(pdf_path)
    out = write_report(pdf_path, report)
    print_summary(report, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path,
                    help="Vendor PDF, e.g. files/<vendor>/<doc>.pdf")
    args = ap.parse_args()
    if not args.pdf.exists():
        ap.error(f"PDF not found: {args.pdf}")
    analyze(args.pdf)


if __name__ == "__main__":
    main()
