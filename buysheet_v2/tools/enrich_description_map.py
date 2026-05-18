"""Enrich vocab/description_map.json with novel product descriptions from extractions.

For each unique product description across one or more `.v2.cards.json` sidecars
that is NOT yet in `description_map.json`, ask Claude (Sonnet 4.6 + Structured
Outputs) to classify it into the template's SG + SSG vocab and store the result.

Cost: ~$0.005 per unique novel description (batched ~30 per call).
Persistence: the map grows over time; each new vendor adds to the cache. Future
runs of the same vendor are free (cache hit).

Usage:
    python -m buysheet_v2.tools.enrich_description_map files/<vendor>/<doc>.v2.cards.json [...]
    python -m buysheet_v2.tools.enrich_description_map --vendor-dir files/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from buysheet_v2.schemas.card import SG, SSG
from buysheet_v2.schemas.extraction_result import CatalogExtraction

REPO = Path(__file__).resolve().parent.parent.parent
VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab" / "description_map.json"

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 30
MAX_TOKENS = 4096


class DescriptionClassification(BaseModel):
    """One Claude classification of a product description."""

    description: str = Field(..., description="The product description being classified, returned verbatim")
    sg: SG = Field(..., description="Sub-group from the template's Product Data SG vocab")
    ssg: SSG = Field(..., description="Sub-sub-group from the template's Product Data SSG vocab")
    confidence: str = Field(
        ...,
        description="One of: high (clear shoe family), medium (plausible inference), low (best guess)",
    )


class ClassificationBatchResponse(BaseModel):
    classifications: list[DescriptionClassification]


SYSTEM_PROMPT = """You are classifying shoe product descriptions into the Kith template's closed vocabulary.

For each description in the user's list, return a `DescriptionClassification` with:
  - description: the original string verbatim
  - sg: one of {Sneakers, Boots, Heels, Miscellaneous, Sandals, Shoes, Slippers}
  - ssg: one of {Basketball, Causal Shoe, Court, Cupsole, Flats, Heels, Miscellaneous, Modern Comfort, Running, Slip On, Sneakerboot, Training, Vulcanized}
  - confidence: 'high' / 'medium' / 'low'

Guidance:
  - Most athletic catalog SKUs are Sneakers
  - 'AIR FORCE 1', 'JORDAN', 'DUNK', 'SAMBA', 'GAZELLE', 'STAN SMITH', 'FORUM', 'CONVERSE CHUCK 70' -> Sneakers / Causal Shoe (Kith template misspells 'Casual' as 'Causal' — use 'Causal Shoe' verbatim)
  - 'AIR MAX', 'PEGASUS', 'ZOOM', 'ADIZERO', 'BOOST' -> Sneakers / Running
  - 'KOBE', 'LEBRON', 'HARDEN', 'TATUM', 'BBALL' -> Sneakers / Basketball
  - 'PREDATOR', 'COPA', 'ULTRA BOOST CLEAT', 'INDOOR', 'SAMBA F50' -> Sneakers / Court (when soccer/football) or Causal Shoe
  - 'TERREX', 'TRAIL', 'HIKE' -> Boots / Modern Comfort (hiking) or Sneakers / Running (trail running)
  - 'LOAFER', 'MOC' -> Shoes / Slip On
  - 'BOOT' in name -> Boots
  - 'SLIDE', 'ADILETTE', 'FLIP' -> Sandals / Slip On
  - 'BALLET', 'FLAT' -> Heels / Flats
  - 'WALKER', 'COMFORT', 'OXFORD' -> Shoes / Modern Comfort or Causal Shoe
  - Use 'Miscellaneous' / 'Miscellaneous' only when truly unclassifiable
  - When unsure between Causal Shoe vs Court for sneakers, prefer Causal Shoe (broader category)

Confidence:
  - high: the silhouette family is unambiguous from the name (AIR FORCE 1, SAMBA OG, CHUCK 70)
  - medium: reasonable inference (e.g. unknown 'AIR ZOOM XYZ' assumed Running)
  - low: name doesn't reveal the shoe family clearly

Return ONE classification per description in the input list, preserving order.
"""


def load_existing_map() -> dict[str, dict]:
    if VOCAB_PATH.exists():
        return json.loads(VOCAB_PATH.read_text())
    return {}


def gather_descriptions_from_sidecars(sidecar_paths: list[Path]) -> set[str]:
    """Collect every non-null product description across the given sidecars."""
    descs: set[str] = set()
    for p in sidecar_paths:
        if not p.exists():
            print(f"  [skip] {p} (not found)", file=sys.stderr)
            continue
        try:
            ext = CatalogExtraction.model_validate_json(p.read_text())
        except Exception as e:
            print(f"  [skip] {p} ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        for c in ext.all_cards:
            if c.description and c.description.strip():
                descs.add(c.description.strip())
    return descs


def classify_batch(client: anthropic.Anthropic, descriptions: list[str]) -> tuple[list[DescriptionClassification], dict]:
    """One Claude call to classify up to BATCH_SIZE descriptions."""
    user_text = (
        f"Classify these {len(descriptions)} shoe product descriptions:\n\n"
        + "\n".join(f"{i + 1}. {d!r}" for i, d in enumerate(descriptions))
    )
    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_text}],
        output_format=ClassificationBatchResponse,
    )
    parsed = response.parsed_output
    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    }
    return parsed.classifications, usage


def enrich(sidecar_paths: list[Path], *, dry_run: bool = False) -> int:
    current = load_existing_map()
    print(f"[enrich] current map has {len(current)} entries")

    all_descs = gather_descriptions_from_sidecars(sidecar_paths)
    print(f"[enrich] gathered {len(all_descs)} unique descriptions across {len(sidecar_paths)} sidecars")

    novel = sorted(d for d in all_descs if d not in current)
    if not novel:
        print("[enrich] nothing to add; all descriptions already in map")
        return 0
    print(f"[enrich] novel descriptions to classify: {len(novel)}")

    if dry_run:
        for d in novel[:20]:
            print(f"  would classify: {d!r}")
        if len(novel) > 20:
            print(f"  ... and {len(novel) - 20} more")
        return 0

    client = anthropic.Anthropic()
    total_added = 0
    total_in = total_out = total_cache = 0
    for batch_start in range(0, len(novel), BATCH_SIZE):
        batch = novel[batch_start:batch_start + BATCH_SIZE]
        print(f"[enrich] classifying batch {batch_start // BATCH_SIZE + 1} "
              f"({len(batch)} descriptions, total {batch_start + len(batch)} / {len(novel)})...")
        try:
            classifications, usage = classify_batch(client, batch)
        except Exception as e:
            print(f"  [batch failed] {type(e).__name__}: {e}", file=sys.stderr)
            continue
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]
        total_cache += usage["cache_read_tokens"]
        # Merge into map
        added_this_batch = 0
        for cls in classifications:
            current[cls.description] = {
                "sg": cls.sg,
                "ssg": cls.ssg,
                "confidence": cls.confidence,
            }
            added_this_batch += 1
        total_added += added_this_batch

    # Save updated map
    VOCAB_PATH.write_text(json.dumps(current, indent=2, sort_keys=True))
    sonnet_in_rate = 3.0  # $/MTok
    sonnet_out_rate = 15.0
    cost = (total_in / 1e6) * sonnet_in_rate + (total_out / 1e6) * sonnet_out_rate
    print(f"\n[enrich] DONE  added={total_added}  total_map_size={len(current)}")
    print(f"  tokens: in={total_in}  out={total_out}  cache_read={total_cache}")
    print(f"  estimated cost: ${cost:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="enrich_description_map")
    ap.add_argument("sidecars", nargs="*", type=Path,
                    help="One or more <doc>.v2.cards.json sidecar paths")
    ap.add_argument("--vendor-dir", type=Path, default=None,
                    help="Scan a directory tree for *.v2.cards.json files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print novel descriptions but skip the Claude calls")
    args = ap.parse_args()

    paths: list[Path] = list(args.sidecars)
    if args.vendor_dir:
        paths.extend(sorted(args.vendor_dir.rglob("*.v2.cards.json")))
    if not paths:
        print("usage: provide sidecar paths or --vendor-dir DIR", file=sys.stderr)
        return 64
    return enrich(paths, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
