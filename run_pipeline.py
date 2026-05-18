"""One-command orchestrator for the agentic pipeline.

Runs classify_pages → extract_products → build_buysheet on a single PDF.
Each step is a function imported from its own module — no subprocess
overhead, same logging/output as running them individually.

Usage:
  python agentic_parsing/run_pipeline.py <pdf>
  python agentic_parsing/run_pipeline.py <pdf> --workers 8
  python agentic_parsing/run_pipeline.py <pdf> --model sonnet
  python agentic_parsing/run_pipeline.py <pdf> --force --out my.xlsx

  # Skip steps already done:
  python agentic_parsing/run_pipeline.py <pdf> --skip classify
  python agentic_parsing/run_pipeline.py <pdf> --skip classify extract
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_buysheet import build, default_out_path  # noqa: E402
from classify_pages import classify_pdf  # noqa: E402
from extract_products import MODEL_OPUS, MODEL_SONNET, extract_pdf  # noqa: E402

STEPS = ("classify", "extract", "build")


def _peek_vendor(pdf_path: Path) -> str | None:
    pages_dir = pdf_path.with_name(pdf_path.stem + ".pages")
    if not pages_dir.exists():
        return None
    for jf in sorted(pages_dir.glob("*.json")):
        ctx = json.loads(jf.read_text()).get("context_after") or {}
        if ctx.get("vendor"):
            return ctx["vendor"]
    return None


def run(pdf_path: Path, *, workers: int, force: bool, model: str,
        out: Path | None, skip: set[str]) -> None:
    started = time.time()

    if "classify" in skip:
        print("=== classify: SKIPPED ===", file=sys.stderr)
    else:
        print("=== classify ===", file=sys.stderr)
        t0 = time.time()
        classify_pdf(pdf_path, only_page=None, force=force, workers=workers)
        print(f"  classify took {time.time() - t0:.1f}s", file=sys.stderr)

    if "extract" in skip:
        print("=== extract: SKIPPED ===", file=sys.stderr)
    else:
        print("=== extract ===", file=sys.stderr)
        t0 = time.time()
        mdl = MODEL_OPUS if model == "opus" else MODEL_SONNET
        extract_pdf(pdf_path, only_page=None, force=force, workers=workers,
                    model=mdl)
        print(f"  extract took {time.time() - t0:.1f}s", file=sys.stderr)

    if "build" in skip:
        print("=== build: SKIPPED ===", file=sys.stderr)
        return

    print("=== build ===", file=sys.stderr)
    t0 = time.time()
    out_path = out or default_out_path(pdf_path, _peek_vendor(pdf_path))
    build(pdf_path, out_path, workers)
    print(f"  build took {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"\nTOTAL wall time: {time.time() - started:.1f}s", file=sys.stderr)
    print(f"OUTPUT: {out_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the full agentic pipeline on one PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("pdf", type=Path, help="Vendor PDF, e.g. files/<vendor>/<doc>.pdf")
    ap.add_argument("--workers", type=int, default=5,
                    help="Parallel workers per step (default 5)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run pages even if their JSONs already exist")
    ap.add_argument("--model", choices=("opus", "sonnet"), default="opus",
                    help="Model for extract stage (default opus)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output xlsx path (default: BUYSHEET_<vendor>.xlsx at repo root)")
    ap.add_argument("--skip", choices=STEPS, nargs="+", default=[],
                    help="Skip one or more steps (e.g. --skip classify extract)")
    args = ap.parse_args()
    if not args.pdf.exists():
        ap.error(f"PDF not found: {args.pdf}")
    run(args.pdf, workers=args.workers, force=args.force,
        model=args.model, out=args.out, skip=set(args.skip))


if __name__ == "__main__":
    main()
