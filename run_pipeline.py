"""User-facing CLI entry point — thin shim over the worker.

Kept for muscle memory and stable naming; the actual orchestration
lives in worker.run_multi (single source of truth for the pipeline,
including both single-doc and multi-doc paths). Calling this file is
identical to invoking `python -m worker.run_multi` with the same args.

Usage:
  python run_pipeline.py <doc.pdf>
  python run_pipeline.py <doc1.pdf> <doc2.xlsx> [...]  # multi-doc
  python run_pipeline.py <doc.pdf> --workers 8 --model opus
  python run_pipeline.py <doc.pdf> --vendor NIKE --out merged.xlsx
  python run_pipeline.py <doc.pdf> --skip-analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from worker.run_multi import main  # noqa: E402


if __name__ == "__main__":
    main()
