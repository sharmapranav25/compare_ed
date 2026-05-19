"""Worker — the single CLI entry point for the buy-sheet pipeline.

Handles both single-doc and multi-doc inputs uniformly: N=1 is just the
trivial case of N≥1. PDF goes through classify+extract (existing path);
Excel reads deterministically via openpyxl (no LLM). Merge joins by
canonicalized SKU with first-non-empty-wins per field in priority order
(leftmost CLI arg = highest priority). Build reuses build_buysheet
helpers without modification.

Per-job worker logic: takes N input docs for one vendor, produces one
buy-sheet xlsx. A future cross-job scheduler (separate layer) will
dispatch many of these runs across many users.

Sequential across docs (vocab cache warms, rate limits stay safe, logs
stay readable). Parallel inside each doc via --workers.

Vendor policy:
  N=1 (single doc)  — lenient: use detected vendor, else filename slug.
  N≥2 (multi doc)   — strict: every doc must declare a vendor (detected
                      or filename-inferred), and detected vendors must
                      agree. --vendor <NAME> bypasses both rules and
                      transfers responsibility to the user.

Usage:
  python -m worker.run_multi files/converse_HO26.pdf
  python -m worker.run_multi files/nike/pricing.xlsx files/nike/catalog.pdf
  python -m worker.run_multi a.pdf b.xlsx --workers 8
  python -m worker.run_multi a.pdf b.xlsx --vendor NIKE --out merged.xlsx
  python -m worker.run_multi a.pdf --skip-analysis
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.fill_rate import analyze  # noqa: E402
from build_buysheet import default_out_path  # noqa: E402
from worker.build_merged import build_merged  # noqa: E402
from worker.formats import excel_adapter, pdf_adapter  # noqa: E402
from worker.formats.excel_adapter import _slug_vendor_from_filename  # noqa: E402
from worker.merge import merge_docs  # noqa: E402

PDF_EXTS = {".pdf"}
EXCEL_EXTS = {".xlsx", ".xlsm"}


def _normalize_one(doc: Path, *, workers: int, force: bool, model: str,
                   vendor_override: str | None,
                   detect_images: bool = False) -> Path:
    ext = doc.suffix.lower()
    if ext in PDF_EXTS:
        print(f"--- normalize PDF {doc.name} ---", file=sys.stderr)
        return pdf_adapter.normalize(doc, workers=workers, force=force,
                                     model=model, detect_images=detect_images)
    if ext in EXCEL_EXTS:
        print(f"--- normalize Excel {doc.name} ---", file=sys.stderr)
        return excel_adapter.normalize(doc, vendor=vendor_override, force=force)
    raise ValueError(f"unsupported file type {ext!r} for {doc}")


def _reconcile_vendor(docs: list[Path], vendors_per_doc: list[str | None],
                      override: str | None) -> str:
    """Resolve a single vendor label for the run.

    --vendor wins absolutely. Otherwise:
      N=1: lenient — detected vendor, else filename slug. Single-doc has
           no cross-doc pollution risk so it shouldn't require --vendor.
      N≥2: strict — every doc must declare a vendor; detected vendors
           must agree; missing-vendor errors before disagreement so the
           message points at the right doc.
    """
    if override:
        return override.strip().upper()

    if len(docs) == 1:
        v = vendors_per_doc[0]
        if v and v.strip():
            return v.strip().upper()
        return _slug_vendor_from_filename(docs[0].stem)

    missing = [d for d, v in zip(docs, vendors_per_doc)
               if not (v and v.strip())]
    if missing:
        names = ", ".join(d.name for d in missing)
        raise SystemExit(
            f"\nERROR: no vendor detected for: {names}\n"
            f"  PDFs need a brand_name cover page for auto-detection;\n"
            f"  Excel docs infer from filename. Pass --vendor <NAME> to\n"
            f"  declare a single label and proceed, or split the run.\n"
        )

    distinct = sorted({v.strip().upper() for v in vendors_per_doc})
    if len(distinct) > 1:
        per_doc = "\n".join(
            f"    {d.name}: {v}" for d, v in zip(docs, vendors_per_doc)
        )
        raise SystemExit(
            f"\nERROR: docs span multiple vendors {distinct}:\n{per_doc}\n"
            f"  Multi-vendor splitting is not in milestone 1. To proceed:\n"
            f"    1. Run each vendor's docs separately, OR\n"
            f"    2. Pass --vendor <NAME> to force a single label.\n"
        )
    return distinct[0]


def run(docs: list[Path], *, workers: int, force: bool, model: str,
        out: Path | None, vendor_override: str | None,
        skip_analysis: bool = False) -> Path:
    started = time.time()

    # Footwear detection runs only on single-doc runs. Multi-doc image
    # merging (which crop wins when two docs both have one for the same
    # SKU) is out of scope, so we skip the whole step when N>=2.
    detect_images = (len(docs) == 1)

    pages_dirs: list[Path] = []
    for doc in docs:
        t0 = time.time()
        pd = _normalize_one(doc, workers=workers, force=force,
                            model=model, vendor_override=vendor_override,
                            detect_images=detect_images)
        pages_dirs.append(pd)
        print(f"  {doc.name}: normalize took {time.time() - t0:.1f}s",
              file=sys.stderr)

    print("=== merge ===", file=sys.stderr)
    products_rows, page_records, vendors_per_doc, sku_conflicts = \
        merge_docs(docs, pages_dirs)
    vendor = _reconcile_vendor(docs, vendors_per_doc, vendor_override)
    per_doc_signals = ", ".join(
        f"{d.name}={v or '?'}" for d, v in zip(docs, vendors_per_doc)
    )
    print(f"  vendor signals per doc: {per_doc_signals}", file=sys.stderr)
    print(f"  resolved vendor: {vendor}", file=sys.stderr)
    print(f"  merged {len(products_rows)} rows from {len(docs)} docs "
          f"({len(sku_conflicts)} SKU conflicts kept separate)",
          file=sys.stderr)

    print("=== build ===", file=sys.stderr)
    out_path = out or default_out_path(docs[0], vendor)
    build_merged(products_rows, page_records, sku_conflicts,
                 vendor, out_path, workers)

    if skip_analysis:
        print("=== analysis: SKIPPED ===", file=sys.stderr)
    else:
        print("=== analysis ===", file=sys.stderr)
        for doc in docs:
            t0 = time.time()
            try:
                analyze(doc)
            except FileNotFoundError as exc:
                # Excel docs that produced 00.json still satisfy analyze's
                # .pages/ existence check; only a truly missing pages dir
                # would land here, which means normalize itself failed.
                print(f"  {doc.name}: analysis skipped ({exc})", file=sys.stderr)
                continue
            print(f"  {doc.name}: analysis took {time.time() - t0:.2f}s",
                  file=sys.stderr)

    print("", file=sys.stderr)
    print(f"TOTAL wall time: {time.time() - started:.1f}s", file=sys.stderr)
    print(f"OUTPUT: {out_path}", file=sys.stderr)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Buy-sheet pipeline (single or multi-doc).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Order of docs on the command line IS the priority order:\n"
               "  leftmost = highest priority; first non-empty value per\n"
               "  field wins during merge. Single-doc runs are the\n"
               "  trivial case (no merge ambiguity).",
    )
    ap.add_argument("docs", type=Path, nargs="+",
                    help="One or more vendor docs (.pdf or .xlsx/.xlsm)")
    ap.add_argument("--workers", type=int, default=5,
                    help="Parallel workers per doc (default 5)")
    ap.add_argument("--force", action="store_true",
                    help="Re-normalize pages even if cached JSON exists")
    ap.add_argument("--model", choices=("opus", "sonnet"), default="opus",
                    help="Model for PDF extract stage (default opus)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output xlsx (default: BUYSHEET_<vendor>.xlsx at repo root)")
    ap.add_argument("--vendor", default=None,
                    help="Override vendor label (used for Excel inference + xlsx)")
    ap.add_argument("--skip-analysis", action="store_true",
                    help="Skip the per-doc fill-rate analysis pass")
    args = ap.parse_args()

    for d in args.docs:
        if not d.exists():
            ap.error(f"doc not found: {d}")
        if d.suffix.lower() not in PDF_EXTS | EXCEL_EXTS:
            ap.error(f"unsupported file type {d.suffix!r} for {d} "
                     f"(milestone 1 supports {sorted(PDF_EXTS | EXCEL_EXTS)})")

    run(args.docs, workers=args.workers, force=args.force,
        model=args.model, out=args.out, vendor_override=args.vendor,
        skip_analysis=args.skip_analysis)


if __name__ == "__main__":
    main()
