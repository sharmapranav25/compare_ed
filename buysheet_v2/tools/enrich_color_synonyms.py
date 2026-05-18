"""Enrich vocab/color_synonyms.json with novel vendor color names.

Walks one or more `.v2.cards.json` sidecars, finds raw vendor colors whose
first-slash token isn't yet mapped in color_synonyms.json, and asks Claude
to classify each novel token into the template's closed StandardColor
vocab. Persists results into color_synonyms.json so future runs of any
catalog that reuses those color names verify standard_color cleanly.

This is the same pattern as enrich_description_map.py — call once after a
cold-vendor extraction (or wire into pipeline.py for auto-enrich) to grow
the color vocab over time at a few cents per ~50 novel colors.

Usage:
    python -m buysheet_v2.tools.enrich_color_synonyms files/<vendor>/<doc>.v2.cards.json [...]
    python -m buysheet_v2.tools.enrich_color_synonyms --vendor-dir files/
    python -m buysheet_v2.tools.enrich_color_synonyms ~/buysheet_runs --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from buysheet_v2.schemas.card import StandardColor
from buysheet_v2.schemas.extraction_result import CatalogExtraction

VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocab" / "color_synonyms.json"

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 50
MAX_TOKENS = 4096


class ColorMapping(BaseModel):
    """One Claude mapping from a raw vendor color name to a canonical."""

    raw: str = Field(..., description="The raw vendor color token, returned verbatim")
    standard: StandardColor = Field(
        ..., description="Canonical color from the Kith template's closed vocab",
    )


class ColorMappingBatchResponse(BaseModel):
    mappings: list[ColorMapping]


SYSTEM_PROMPT = """You are mapping vendor shoe-color names to the Kith template's closed StandardColor vocab.

For each color token in the user's list, return a `ColorMapping` with:
  - raw: the original token verbatim
  - standard: one of {Beige, Black, Blue, Brown, Burgundy, Clear, Gold, Gold/Silver, Green, Grey, Multi, Orange, Pink, Purple, Red, White, Yellow}

Guidance:
  - Vendor names are often metaphorical/seasonal: "STARDUST" -> Grey, "ELEPHANT GREY" -> Grey, "ALABASTER" -> White, "SESAME" -> Beige, "OAT MILK" -> Beige
  - "STORMY SKIES" -> Blue, "COLD PLUNGE BLUE" -> Blue, "VARSITY NAVY" -> Blue, "FADED NAVY" -> Blue
  - "DRIZZLE", "THUNDER CLOUD" -> Grey
  - "ROSE CREAM", "DEWY PINK", "LILAC HAZE" -> Pink (lilac is on the pink-purple line; lean Pink unless explicitly purple)
  - "SILVER LAVENDER", "DRIED LAVENDER" -> Purple
  - "WHISKEY", "CHESTNUT", "CAMEL", "RICH MAHOGANY" -> Brown
  - "MILK TEA", "CASHEW BUTTER" -> Beige
  - "NEON LEMON ZEST" -> Yellow
  - "BLACK CHERRY" -> Burgundy (it's a deep red-purple, not Red)
  - "WORK BLUE", "GAME ROYAL", "DEEP ROYAL BLUE" -> Blue
  - "BRT CACTUS", "PACIFIC MOSS", "SCENERY GREEN" -> Green
  - "PHANTOM", "SHADOW BROWN" -> Grey / Brown (use Brown when the modifier is brown, else Grey)
  - "BAROQUE BROWN", "MED BROWN", "LT ARMORY BLUE" — strip the modifier, the head noun is the color family
  - "SAFETY ORANGE", "MESA ORANGE", "CAMPFIRE ORANGE" -> Orange
  - "SUNDIAL", "OASIS TEAL" -> Yellow / Blue respectively (teal leans Blue in the Kith vocab since there's no Teal)
  - Abbreviations: "DK" = dark, "LT" = light, "MTLC" = metallic, "BRT" = bright, "WMNS" = women's (ignore)
  - When a token is clearly a metallic finish (gold, silver), use Gold/Silver
  - When you genuinely can't infer, use Multi (NOT Miscellaneous) — Multi is the catch-all for multi-tone or unclassifiable
  - DON'T map two-tone compound strings — the input list contains single color tokens only

Return ONE mapping per token in the input list, preserving order. Never invent tokens that weren't in the input."""


def load_existing_map() -> dict[str, str]:
    if VOCAB_PATH.exists():
        return json.loads(VOCAB_PATH.read_text())
    return {}


def _first_slash_token(raw_color: str) -> str:
    """Return the lookup key the oracle uses for a vendor raw color string.

    Matches _lookup_standard_color: take everything before the first `/`,
    lowercased and stripped. This is the granularity we enrich at — the
    next-tier first-word fallback in the oracle hits if this token resolves.
    """
    return raw_color.lower().split("/")[0].strip()


def gather_novel_tokens(
    sidecar_paths: list[Path], existing: dict[str, str],
) -> list[str]:
    """Collect raw-color first-slash tokens that aren't yet in the synonym map."""
    novel: set[str] = set()
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
            if not c.color or not c.color.strip():
                continue
            token = _first_slash_token(c.color)
            if not token:
                continue
            if token in existing:
                continue
            # Also try the bare first word — if THAT resolves the oracle's
            # third fallback will fire, so we don't need to enrich the token.
            first_word = token.split()[0]
            if first_word and first_word in existing:
                continue
            novel.add(token)
    return sorted(novel)


def classify_batch(
    client: anthropic.Anthropic, tokens: list[str],
) -> tuple[list[ColorMapping], dict]:
    """One Claude call to classify up to BATCH_SIZE color tokens."""
    user_text = (
        f"Map these {len(tokens)} vendor color names to the Kith template's "
        f"StandardColor vocab:\n\n"
        + "\n".join(f"{i + 1}. {t!r}" for i, t in enumerate(tokens))
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
        output_format=ColorMappingBatchResponse,
    )
    parsed = response.parsed_output
    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", 0),
        "output_tokens": getattr(response.usage, "output_tokens", 0),
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
    }
    return parsed.mappings, usage


def enrich(sidecar_paths: list[Path], *, dry_run: bool = False) -> int:
    current = load_existing_map()
    print(f"[enrich] current color_synonyms.json has {len(current)} entries")

    novel = gather_novel_tokens(sidecar_paths, current)
    if not novel:
        print("[enrich] nothing to add; every vendor color resolves via existing vocab")
        return 0
    print(f"[enrich] novel color tokens to classify: {len(novel)} "
          f"(across {len(sidecar_paths)} sidecars)")

    if dry_run:
        for t in novel[:30]:
            print(f"  would classify: {t!r}")
        if len(novel) > 30:
            print(f"  ... and {len(novel) - 30} more")
        return 0

    client = anthropic.Anthropic()
    total_added = 0
    total_in = total_out = total_cache = 0
    for batch_start in range(0, len(novel), BATCH_SIZE):
        batch = novel[batch_start:batch_start + BATCH_SIZE]
        print(f"[enrich] classifying batch {batch_start // BATCH_SIZE + 1} "
              f"({len(batch)} tokens, total {batch_start + len(batch)} / {len(novel)})...")
        try:
            mappings, usage = classify_batch(client, batch)
        except Exception as e:
            print(f"  [batch failed] {type(e).__name__}: {e}", file=sys.stderr)
            continue
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]
        total_cache += usage["cache_read_tokens"]
        # Merge — keys are lowercase already (Claude returns the raw verbatim,
        # but our oracle lookups lowercase before checking, so be defensive).
        for m in mappings:
            current[m.raw.lower().strip()] = m.standard
            total_added += 1

    VOCAB_PATH.write_text(json.dumps(current, indent=2, sort_keys=True))
    sonnet_in_rate = 3.0  # $/MTok
    sonnet_out_rate = 15.0
    cost = (total_in / 1e6) * sonnet_in_rate + (total_out / 1e6) * sonnet_out_rate
    print(f"\n[enrich] DONE  added={total_added}  total_map_size={len(current)}")
    print(f"  tokens: in={total_in}  out={total_out}  cache_read={total_cache}")
    print(f"  estimated cost: ${cost:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="enrich_color_synonyms")
    ap.add_argument("sidecars", nargs="*", type=Path,
                    help="One or more <doc>.v2.cards.json sidecar paths")
    ap.add_argument("--vendor-dir", type=Path, default=None,
                    help="Scan a directory tree for *.v2.cards.json files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print novel color tokens but skip the Claude calls")
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
