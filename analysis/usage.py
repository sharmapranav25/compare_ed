"""Token-usage capture + USD cost estimation for every LLM call.

Each LLM call site extracts `resp.usage` via `usage_from_response()`,
which returns a small dict. Those dicts are stored:

  - Per-page (classify / extract / validate)  → in <NN>.json under "usage"
  - Per-build (vocab_map) — aggregated across all (field, key) lookups
                                        → in <pages>/_build_usage.json

`compute_fill_rate()` in fill_rate.py reads both, sums by model and
multiplies through `PRICING` for an estimated USD spend per step.

PRICING reflects Anthropic's published rates (USD per million tokens).
Edit this table if the rates change; everything downstream re-reads it.
"""

from __future__ import annotations

from typing import Any

# USD per million tokens. Standard published rates for Claude 4.x family.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input":  3.00, "output": 15.00},
}

# Anthropic prompt-cache multipliers vs. base input price.
CACHE_READ_MULT  = 0.10   # cached input is billed at 10% of base
CACHE_WRITE_MULT = 1.25   # 5-min TTL cache creation (ephemeral)

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def empty_usage(model: str | None = None) -> dict:
    out: dict[str, Any] = {f: 0 for f in USAGE_FIELDS}
    out["n_calls"] = 0
    if model is not None:
        out["model"] = model
    return out


def usage_from_response(resp, model: str) -> dict:
    """Extract token counts off an anthropic.messages.Message response."""
    u = getattr(resp, "usage", None)
    out = empty_usage(model)
    out["n_calls"] = 1
    if u is None:
        return out
    for f in USAGE_FIELDS:
        out[f] = int(getattr(u, f, 0) or 0)
    return out


def add_usage(acc: dict, more: dict) -> dict:
    """Sum `more` into `acc` in place. Both must share `model` (or `acc`
    starts empty and inherits). Returns `acc` for chaining."""
    if not more:
        return acc
    if "model" not in acc and "model" in more:
        acc["model"] = more["model"]
    for f in USAGE_FIELDS:
        acc[f] = int(acc.get(f, 0)) + int(more.get(f, 0) or 0)
    acc["n_calls"] = int(acc.get("n_calls", 0)) + int(more.get("n_calls", 1))
    return acc


def cost_usd(entry: dict) -> float:
    """USD estimate for one usage entry. Returns 0.0 if the model isn't
    in PRICING (so an unknown model never crashes the report)."""
    if not entry:
        return 0.0
    model = entry.get("model")
    if not model or model not in PRICING:
        return 0.0
    p = PRICING[model]
    base_in = p["input"]  / 1_000_000
    base_out = p["output"] / 1_000_000
    in_t  = int(entry.get("input_tokens", 0) or 0)
    out_t = int(entry.get("output_tokens", 0) or 0)
    cr_t  = int(entry.get("cache_read_input_tokens", 0) or 0)
    cw_t  = int(entry.get("cache_creation_input_tokens", 0) or 0)
    return (in_t * base_in
            + out_t * base_out
            + cr_t  * base_in * CACHE_READ_MULT
            + cw_t  * base_in * CACHE_WRITE_MULT)


def summarize(entries: list[dict]) -> dict:
    """Group a list of usage entries by model and aggregate. Used by the
    fill-rate report. Returns {model: {tokens..., n_calls, cost_usd}}.
    """
    by_model: dict[str, dict] = {}
    for e in entries:
        if not e:
            continue
        m = e.get("model") or "unknown"
        acc = by_model.setdefault(m, empty_usage(m))
        add_usage(acc, e)
    for m, acc in by_model.items():
        acc["cost_usd"] = round(cost_usd(acc), 4)
    return by_model
