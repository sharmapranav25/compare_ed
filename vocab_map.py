"""LLM-driven vendor → buy-sheet dropdown vocab mapper.

Pure function in spirit: caller passes a closed dropdown list + the vendor
signals it has, and gets back a chosen canonical value or None.

Returns
  {"value": <one of choices> | None, "confidence": "high"|"medium"|"low",
   "source": "cache" | "llm"}

Resolution policy:
  1. Cache hit (keyed by field + normalized primary signal) → return
  2. Sonnet call with system prompt that lists the field + choices + signals
  3. Result cached. low-confidence is treated the same as None (force unmapped).

Cache files live at agentic_parsing/cache/<field_slug>.json so the next run
costs $0 for any previously-seen vendor value (cross-document cache hits).

Importable API:
  load_cache(field) -> dict
  save_cache(field, cache) -> None
  map_to_dropdown(*, field, choices, key, signals, client, cache) -> dict

CLI for poking:
  python agentic_parsing/vocab_map.py --field "SG" \\
      --choices "Sneakers,Boots,Heels,Miscellaneous,Sandals,Shoes,Slippers" \\
      --key "STAN SMITH" --signal description=STAN_SMITH section=MENS_LIFESTYLE
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.usage import usage_from_response  # noqa: E402

load_dotenv()

MODEL = "claude-sonnet-4-6"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

SYSTEM = """You map vendor wording from a wholesale shoe buy-sheet PDF to a closed dropdown vocabulary used in our internal spreadsheet.

You will be given:
  - The buy-sheet field being mapped (e.g. MG, SG, SSG, Standard Color, INTRO DATE)
  - The closed list of allowed dropdown values for that field
  - A small set of signals pulled from the page (description, section, SKU,
    gender hints, etc.) — use all of them together

Return JSON ONLY (no prose, no code fences):
  {"value": "<one of choices>" | null, "confidence": "high" | "medium" | "low"}

The two output dimensions mean different things — keep them independent:

  value:
    - one of the listed choices = that choice is a real fit
    - null                      = none of the listed choices fits this vendor value
                                  (downstream will write the raw vendor value instead)

  confidence:
    - high   = the signals clearly identify the answer (whether it's a choice
               or a confident null)
    - medium = mostly clear, minor ambiguity
    - low    = you are guessing; the signals don't tell you enough
               (downstream will LEAVE THE CELL EMPTY when confidence is low)

So: choose null only when you can confidently say "no choice in the list
applies." If you genuinely don't know whether something fits, return low
confidence instead — that's the "leave the cell empty" signal, not null.

Conventions:
  - MG: M-Footwear = men's, W-Footwear = women's, K-Footwear = kids / youth /
    grade school / unisex-leaning-kids. Suffixes/prefixes like "W ", "WMNS",
    "GS", "TD", "K-" are strong signals.
  - SG / SSG: silhouette / model name in the description is the strongest
    signal (STAN SMITH / SAMBA / SUPERSTAR → court Sneakers; AIR MAX → running
    Sneakers; CLOG → Sandals; etc.). The section header is the next-strongest.
  - Standard Color: the vendor color is often multi-tone (e.g.
    "core black/core white/cream white"). Pick the dominant first-listed tone.
  - INTRO DATE: vendor dates come as "5/1/25", "JUL 2025", "2025-07-15", etc.
    Map to the month abbreviation (JAN..DEC) using the month component. Treat
    US-style M/D/YY as month-first.
"""


def _slug(field: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_") or "field"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load_cache(field: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{_slug(field)}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_cache(field: str, cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{_slug(field)}.json"
    p.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _parse_json(raw: str) -> dict:
    """Tolerant: never raises. Returns {} on any parse failure so the
    caller's downstream logic treats the response as low-confidence
    (cell-empty) instead of crashing the whole build's thread pool."""
    raw = raw.strip()
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass
    s = raw.find("{")
    if s == -1:
        return {}
    # raw_decode parses one JSON object and ignores trailing junk —
    # handles the case where Sonnet returns {valid}{garbage} despite
    # "JSON only" in the prompt.
    try:
        obj, _end = json.JSONDecoder().raw_decode(raw[s:])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _format_signals(signals: dict) -> str:
    if not signals:
        return "  (none provided)"
    lines = []
    for k, v in signals.items():
        if v is None or v == "":
            continue
        lines.append(f"  {k}: {v}")
    return "\n".join(lines) if lines else "  (all empty)"


def map_to_dropdown(
    *,
    field: str,
    choices: list[str],
    key: str,
    signals: dict,
    client: anthropic.Anthropic,
    cache: dict,
) -> dict:
    """Map vendor signals to one of `choices` for the named buy-sheet field.

    Args:
      field   : buy-sheet field name ("MG", "SG", "SSG", "Standard Color", ...)
      choices : the closed dropdown vocabulary, in order
      key     : cache key — caller picks the primary signal to dedupe on
                (e.g. description for SG/SSG, color for Standard Color)
      signals : all signals to feed the LLM (description, section, sku,
                gender_hint, color, ...)
      client  : a configured Anthropic client
      cache   : the cache dict for this field (mutated in place on a miss)

    Returns:
      {"value": <choice|None>, "confidence": "high"|"medium"|"low", "source": "cache"|"llm"}
    """
    norm_key = _norm(key)
    if not norm_key:
        return {"value": None, "confidence": "low", "source": "llm"}
    if norm_key in cache:
        hit = dict(cache[norm_key])
        hit["source"] = "cache"
        return hit

    prompt = (
        f"Field: {field}\n"
        f"Choices: {json.dumps(choices)}\n"
        f"Signals:\n{_format_signals(signals)}\n\n"
        "Return JSON only."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=128,
        system=[{"type": "text", "text": SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    usage = usage_from_response(resp, MODEL)
    parsed = _parse_json(resp.content[0].text)
    value = parsed.get("value")
    confidence = parsed.get("confidence", "low")
    if value is not None and value not in choices:
        value = None
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    result = {"value": value, "confidence": confidence}
    # Cache only verdicts we're confident about. A "low" result means the
    # signals were insufficient — a future occurrence with different signals
    # deserves a fresh attempt, so don't bake "we didn't know" into the cache.
    if confidence != "low":
        cache[norm_key] = result
    out = dict(result)
    out["source"] = "llm"
    out["usage"] = usage
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--choices", required=True, help="Comma-separated dropdown choices")
    ap.add_argument("--key", required=True, help="Cache key (primary signal)")
    ap.add_argument("--signal", action="append", default=[],
                    help="Signal as name=value (repeatable)")
    ap.add_argument("--no-cache-save", action="store_true")
    args = ap.parse_args()

    choices = [c.strip() for c in args.choices.split(",") if c.strip()]
    signals: dict = {}
    for s in args.signal:
        if "=" in s:
            k, v = s.split("=", 1)
            signals[k.strip()] = v.strip()
    client = anthropic.Anthropic(max_retries=10)
    cache = load_cache(args.field)
    result = map_to_dropdown(
        field=args.field, choices=choices, key=args.key,
        signals=signals, client=client, cache=cache,
    )
    if not args.no_cache_save:
        save_cache(args.field, cache)
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
