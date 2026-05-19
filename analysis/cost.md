# Cost — what each pipeline run actually billed

Lives in [usage.py](usage.py). Every LLM call in the pipeline records
its token usage on disk; [fill_rate.py](fill_rate.py) sums those into
the per-PDF report and multiplies through the pricing table for a USD
estimate.

## Why this exists

CLAUDE.md quoted a per-page cost range ($0.10–$0.25 per dense product
page). That's useful but not actionable — you can't compare a run today
to a run last week, or see *which step* dominates spend on a given PDF.

With usage capture you get:

- An exact per-step token count after every run.
- A USD estimate based on the published pricing table (estimate, not
  invoice — see Caveats below).
- A persistent on-disk record (`analysis/<pdf-stem>.json`) so you can
  diff runs over time.

## Pricing table (USD per million tokens)

Stored in `PRICING` at the top of [usage.py](usage.py). Edit if Anthropic
changes their schedule.

| model | input | output |
|---|---|---|
| `claude-opus-4-7` | $15.00 | $75.00 |
| `claude-sonnet-4-6` | $3.00 | $15.00 |

Cache multipliers (vs base input price):

| | multiplier | meaning |
|---|---|---|
| `CACHE_READ_MULT` | **0.10** | cache hits billed at 10% of base input |
| `CACHE_WRITE_MULT` | **1.25** | 5-minute TTL ephemeral cache creation billed at 125% of base input |

## Where usage is captured

| LLM call site | Module | Stored as | Bill goes to |
|---|---|---|---|
| Page classification | [classify_pages.py](../classify_pages.py) → `classify_page` | `<NN>.json → usage.classify` | `classify` |
| Stage 1 visual extract | [extract_products.py](../extract_products.py) → `extract_page` | `<NN>.json → usage.extract` | `extract` |
| Stage 2 SKU validator (fallback) | [extract_products.py](../extract_products.py) → `validate_skus` | `<NN>.json → usage.validate` | `validate` |
| Dropdown mapping (vocab) | [vocab_map.py](../vocab_map.py) → `map_to_dropdown` | aggregated to `<pdf-stem>.pages/_build_usage.json` | `vocab_map` |

The per-page JSONs carry per-call entries; the build's vocab_map calls
are aggregated to a single sidecar (one entry per model summing all
calls) because they're per-`(field, key)`-tuple, not per-page.

### Usage entry shape

```json
{
  "model": "claude-opus-4-7",
  "n_calls": 9,
  "input_tokens": 49528,
  "output_tokens": 2127,
  "cache_read_input_tokens": 0,
  "cache_creation_input_tokens": 0
}
```

Aggregation respects model name — if you mix models for the same step
(e.g. running extract with `--model sonnet` once and `--model opus`
another), aggregation is keyed by model. (Today every step has a single
fixed model; this is forward-compatible.)

## Cost formula

For one usage entry:

```
cost_usd = (input_tokens          × base_in)
         + (output_tokens         × base_out)
         + (cache_read_tokens     × base_in × 0.10)
         + (cache_creation_tokens × base_in × 1.25)
```

Where `base_in = PRICING[model]["input"] / 1_000_000`.

The report rounds to 4 decimal places ($0.0001 = ~hundredth of a cent).
This is precise enough to spot ~10× cost regressions, not precise enough
to reconcile against a billing invoice (and shouldn't be used for that).

## Reading a report

Example from Converse (`analysis/converse_HO26.json`):

```
=== fill-rate ===
  pages:        17 total, 9 product, 2 without text layer
  SKU coverage: 15 / 17 candidates  (88.2%)
  ...
  LLM spend (USD, estimate):
    classify   $0.0982  (17 calls, in/out 31846/179)
    extract    $0.9024  (9 calls, in/out 49528/2127)
    validate   $0.0000  (0 calls, in/out 0/0)
    vocab_map  $0.0957  (40 calls, in/out 28436/692)
    total      $1.0963
```

Things to notice:

1. **Opus extract is the dominant cost** (82% on Converse). It's the
   only step paying Opus prices ($15/$75 per MTok). Sonnet steps are 5×
   cheaper at the input rate, 5× cheaper at output.
2. **`validate` is $0.0000.** This is the deterministic check working —
   Stage 2 only fires when the text layer is absent. On Converse, the
   two text-layer-absent pages had null SKUs so even the LLM validator
   short-circuited; on Nike, all pages took the deterministic path.
3. **`vocab_map` doesn't show cache savings yet.** First run on a fresh
   vendor → no cache hits. A second run (same vendor or one that reuses
   dropdown values like "BLACK", "M-Footwear") will show
   `cache_read_input_tokens` > 0 and the cost will drop. The vocab cache
   is at [`cache/<field>.json`](../cache/).

## Cost vs PDF shape

| PDF | pages | product | candidates | extract cost | total | $ / kept row |
|---|---|---|---|---|---|---|
| Nike (early run, pre-usage capture) | 9 | 9 | 112 | ~$? | ~$? | est ~$0.01 |
| Converse | 17 | 9 | 17 | $0.9024 | $1.0963 | $0.073 |

Per-row cost is a useful normalizer when comparing across catalogs —
dense line-sheets (Nike) are much more cost-efficient per SKU than
sparse hero-page catalogs (Converse).

## Caveats — this is an estimate, not an invoice

- **Pricing assumed current.** The `PRICING` table is checked into the
  repo. If Anthropic changes pricing, edit it; the next analysis run
  picks it up.
- **Tokens counted from `resp.usage`.** Reflects what the API server
  reports, but billing reconciliation may differ for promotional credits,
  enterprise contracts, etc.
- **Cache pricing is a multiplier model.** Anthropic's actual cache
  pricing is per-multiplier-per-model with edge cases (1-hour TTL has a
  different multiplier than 5-minute; we assume 5-minute / ephemeral
  since that's what `vocab_map.py` requests). For a precise number,
  check the API console.
- **Failed retries are charged.** The SDK's `max_retries=10` means a
  burst of 5xx errors can cost more than a clean run. Usage capture
  happens after a successful response, so failed retries don't show up
  in `n_calls` (under-counts cost in the rare retry-storm case).

## Extending — usage hooks elsewhere

To capture usage at a new LLM call site:

```python
from analysis.usage import usage_from_response

resp = client.messages.create(model=MODEL, ...)
usage = usage_from_response(resp, MODEL)   # → dict
# store wherever — e.g. record["usage"]["my_step"] = usage
```

The dict integrates automatically if you add the step name to
[`fill_rate.py`'s `_STEPS`](fill_rate.py) tuple and aggregate it in
`compute_fill_rate`.

## Companion docs

- [fill_cells.md](fill_cells.md) — the quality side of the same report
  (what you got).
- [../deterministic_check/deterministic_check.md](../deterministic_check/deterministic_check.md)
  — why `validate` is now usually $0.
