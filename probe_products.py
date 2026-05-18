"""Single-page Opus vision probe — answers 5 product questions about one PDF page.

Renders one page from a PDF at 300 DPI, sends to Claude Opus, prints + saves
the response. PNG + response are kept under
  agentic_parsing/probes/<doc-stem>/<NN>.{png,txt}
for re-inspection.

Usage:
  python agentic_parsing/probe_products.py files/<vendor>/<doc>.pdf 12
  python agentic_parsing/probe_products.py files/<vendor>/<doc>.pdf 12 --force
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import anthropic
import fitz
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MODEL = "claude-opus-4-7"
DPI = 300
PROBES_DIR = Path(__file__).resolve().parent / "probes"

PROMPT = """You are looking at ONE page from a wholesale shoe buy-sheet / vendor line-sheet PDF.

Answer these five questions about THIS PAGE only. Be precise. If something is not visible, say so — do not invent.

1) Do you see shoe products in the image with unique identifiers (SKUs / style codes)?
   Answer yes/no with a one-sentence reason.

2) If yes, list every SKU / unique product identifier visible on the page.
   Preserve exact casing, punctuation, and any letter/number suffix as printed.

3) For each SKU above, give the item description as printed on the page
   (product / silhouette / model name — whatever the vendor wrote).

4) For each SKU, give the color description as written by the brand
   (the color name printed on the page — not your interpretation).

5) For each SKU, give the price(s) as printed (wholesale / cost / MSRP / retail —
   include the label and currency exactly as shown, e.g. "WHSL $65.00", "MSRP $130").
   If multiple price types are shown, list all of them.

Format your answer as:

Q1: <yes/no> — <reason>

Q2/Q3/Q4/Q5 combined per SKU:
  - SKU: <as printed>
    description: <as printed, or "not visible">
    color: <as printed, or "not visible">
    price: <as printed with label + currency, or "not visible">
  - SKU: ...
"""


def render_page_png(pdf: fitz.Document, page_no: int, out_path: Path, dpi: int = DPI) -> None:
    page = pdf[page_no - 1]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(out_path, format="PNG")


def png_to_b64_block(png_path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_path.read_bytes()).decode(),
        },
    }


def probe(pdf_path: Path, page_no: int, force: bool) -> None:
    pdf = fitz.open(str(pdf_path))
    try:
        if not (1 <= page_no <= pdf.page_count):
            raise ValueError(f"page {page_no} out of range 1..{pdf.page_count}")
        probe_dir = PROBES_DIR / pdf_path.stem
        probe_dir.mkdir(parents=True, exist_ok=True)
        png_path = probe_dir / f"{page_no:02d}.png"
        txt_path = probe_dir / f"{page_no:02d}.txt"
        if txt_path.exists() and not force:
            print(f"existing response at {txt_path} (use --force to re-run):\n", file=sys.stderr)
            print(txt_path.read_text())
            return
        if not png_path.exists() or force:
            render_page_png(pdf, page_no, png_path)
        client = anthropic.Anthropic(max_retries=10)
        resp = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            messages=[{
                "role": "user",
                "content": [
                    png_to_b64_block(png_path),
                    {"type": "text", "text": PROMPT},
                ],
            }],
        )
        answer = resp.content[0].text
        txt_path.write_text(answer)
        usage = resp.usage
        print(f"page {page_no:02d} -> {txt_path}  "
              f"(in={usage.input_tokens}, out={usage.output_tokens})\n", file=sys.stderr)
        print(answer)
    finally:
        pdf.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path, help="Vendor PDF")
    ap.add_argument("page", type=int, help="1-indexed page number")
    ap.add_argument("--force", action="store_true", help="Re-run even if response exists")
    args = ap.parse_args()
    if not args.pdf.exists():
        ap.error(f"PDF not found: {args.pdf}")
    probe(args.pdf, args.page, args.force)


if __name__ == "__main__":
    main()
