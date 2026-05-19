# Fill-cells (fill-rate) — what fraction of cells got populated

Lives in [fill_rate.py](fill_rate.py). Pure local compute, no LLM calls.
Reads the page JSONs already written by classify + extract and asks:
**of the rows we kept, how much of each column actually got filled?**

## Why this exists

Before this report there was no way to tell whether the pipeline was
losing fields silently — Opus might emit a row with `cost: null` and we'd
write an empty cell with no log line saying "1 cost cell skipped." With
fill-rate you can spot:

- A vendor catalog where retail is on every page but cost is on none
  → fill rate `cost = 0.0%`, `retail = 100.0%`. (Nike does this.)
- A vendor whose color descriptions don't canonicalize against the text
  layer → 30 unverified colors → REVIEW lights up.
- A drop in SKU coverage between runs → something regressed in the
  pipeline (or the vendor changed PDF formatting).

The metric pairs naturally with [cost.md](cost.md): together they tell
you *what you got vs. what you paid for*.

## Schema (the JSON it writes)

`analysis/<pdf-stem>.json`:

```json
{
  "pdf": "converse_HO26",
  "totals": {
    "pages_total": 17,
    "pages_product": 9,
    "pages_text_layer_absent": 2,
    "candidates": 17,
    "kept": 15,
    "dropped_deterministic": 0,
    "dropped_llm_validator": 0
  },
  "sku_coverage": 0.882,
  "field_fill_rate": {
    "description": 1.0,
    "color":       1.0,
    "cost":        1.0,
    "retail":      1.0,
    "intro_date":  1.0
  },
  "verification_flags": {
    "description_unverified": 3,
    "color_unverified":       0,
    "cost_not_in_text_layer": 0,
    "retail_not_in_text_layer": 0,
    "intro_date_unverified":   0
  },
  "usage":    { /* from cost.md */ },
  "cost_usd": { /* from cost.md */ }
}
```

## Definitions

| metric | definition |
|---|---|
| **`candidates`** | Total Stage-1 Opus emissions across all `product` pages. Sum of each page's `n_candidates`. |
| **`kept`** | Rows that survived all filters and got written to the xlsx. |
| **`dropped_deterministic`** | Rows whose SKU wasn't found in the PDF text layer (see [deterministic_check](../deterministic_check/deterministic_check.md)). |
| **`dropped_llm_validator`** | Rows the Sonnet validator rejected as not-a-SKU. Only fires on pages without a usable text layer. |
| **`sku_coverage`** | `kept / candidates`. The fraction of Opus's emissions that survived QA. |
| **`field_fill_rate[F]`** | For column F, the fraction of `kept` rows whose value is non-null/non-empty. |
| **`verification_flags[F_state]`** | Count of kept rows whose deterministic verifier marked field F as `state` (one of `unverified`, `not_in_text_layer`). |
| **`pages_text_layer_absent`** | Count of pages where PyMuPDF couldn't return readable text. These pages bypass the deterministic check entirely and route to the LLM fallback. |

### SKU coverage caveat: it includes silent drops

When Opus emits a candidate with `sku: null` on a text-layer-absent page,
`validate_skus` short-circuits at [extract_products.py:230](../extract_products.py#L230)
and the candidate disappears with `n_rejected: 0`. The drop doesn't get
attributed to either `dropped_deterministic` or `dropped_llm_validator`
— it just vanishes from `kept`. SKU coverage still catches it:
`kept < candidates` regardless of the path. On Converse this accounts
for `2 / 17` = 11.8% missing coverage.

## Field-by-field semantics

| Field | Counted as filled when… | Verification flag fires when… |
|---|---|---|
| `description` | `_is_filled(p["description"])` (non-empty after strip) | the canonicalized description doesn't appear in the canonicalized text layer |
| `color` | same | same as above for color |
| `cost` | `_is_filled(p["cost"])` after the deterministic check may have cleared it to `None` | deterministic check found the SKU but couldn't substring-match the cost in text layer → value gets nulled and counted as `cost_not_in_text_layer` (this is why `cost: 0.0%` can mean either "vendor doesn't print cost" or "cost was hallucinated and cleared") |
| `retail` | same as `cost` | same |
| `intro_date` | same | same as description (kept, marked unverified) |

`sku` is excluded from `field_fill_rate` — a row without a SKU can't
exist, so it's always 100% by construction.

## Reading the numbers (Nike vs Converse)

| metric | Nike (9 pgs) | Converse (17 pgs) | reading |
|---|---|---|---|
| pages_product | 9 | 9 | Nike is all product; Converse has covers/dividers |
| candidates | 112 | 17 | Nike's pages are dense line-sheets; Converse is sparse hero-spread |
| kept | 112 | 15 | 100% recall on Nike; 88.2% on Converse |
| sku_coverage | 100% | 88.2% | Converse loses 2 to null-SKU on text-layer-absent pages |
| cost fill | 0.0% | 100.0% | Nike prints only retail; Converse prints both |
| description_unverified | 2 | 3 | both are VLM-normalization noise (curly quotes, dashes) — not hallucinations |

## How it's invoked

### Auto, after every `run_pipeline.py`

[run_pipeline.py](../run_pipeline.py) calls `analyze(pdf_path)` after
build. Adds <100 ms to total wall time. Skippable via
`--skip analysis`.

### Standalone

```bash
python -m analysis.fill_rate files/<vendor>/<doc>.pdf
```

Reads the existing `<pdf-stem>.pages/*.json` and `_build_usage.json`
sidecar, recomputes, overwrites `analysis/<pdf-stem>.json`. Safe to
re-run any time — no LLM calls.

## What this does NOT measure

- **Correctness of the value.** If Opus reads `JA1013-010` as `JA1013-100`
  and the swapped SKU happens to appear elsewhere in the text layer, the
  deterministic check passes and fill rate counts it as 100% filled.
  Catching that needs spatial info — out of scope.
- **Quality of the dropdown mapping** (MG/SG/SSG/Standard Color/INTRO
  DATE). Those cells are populated by [vocab_map.py](../vocab_map.py)
  during build. Their fill is implicit in the kept row; we don't track
  per-dropdown confidence separately. (`vocab_map` does write `confidence`
  per resolution — adding a "dropdown low-confidence rate" metric would
  be a sensible extension.)
- **Pages we *should* have classified as product but didn't.** If
  classify mis-labels a product page as `index_or_other`, the page never
  gets extracted and never shows up in `candidates`. Fill rate can't see
  it. Manual REVIEW + classify-prompt iteration is the only mitigation.

## Where the numbers come from (call graph)

```
compute_fill_rate(pdf_path)
  ├─ glob *.pages/*.json (skip _build_usage.json)
  ├─ per record:
  │     pages_*++
  │     candidates += rec["n_candidates"]
  │     kept       += len(rec["products"])
  │     dropped_*  += per rec["rejected_candidates"][i]["stage"]
  │     field_filled[f] += _is_filled(p[f]) for each kept row
  │     flag_counts[f]  += verification[f] == VERIFICATION_FLAGS[f]
  ├─ read _build_usage.json → vocab_map usage
  └─ produce report dict
```

See [usage.py](usage.py) for token aggregation and
[cost.md](cost.md) for the $ side of the same report.
