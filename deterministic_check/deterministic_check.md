# Deterministic text-layer verification

Lives in [verify.py](verify.py). A zero-cost, evidence-based filter that
runs between Stage 1 (Opus visual extract) and Stage 2 (Sonnet SKU
validator) in [extract_products.py](../extract_products.py). It checks
each field the VLM emitted against the PDF's actual text layer and
either drops, clears, or flags rows based on per-field policy.

## Why this exists

The original Stage 2 SKU validator is itself an LLM — a "vibes check"
with no actual evidence. It lets two failure modes through silently:

1. **Character hallucination.** Opus reads `A24021` as `A24021G`, or
   `$65` as `$95`. The Sonnet validator can't tell.
2. **Invented fields.** VLM emits a description / color that wasn't
   physically on the page.

Most catalog PDFs ship a real text layer that PyMuPDF reads in one
zero-cost call. Use it as ground truth: any string the VLM claims should
be physically present in the text layer (after whitespace + case
canonicalization).

The deterministic check is precision-floor — it can't add SKUs the VLM
missed, but it can drop ones the VLM invented.

## The single helper that does the work

```python
def canon(s: str | None) -> str:
    return re.sub(r"\s+", "", s or "").upper()
```

That's it. Strip every whitespace character, uppercase everything. So:

| input | canon |
|---|---|
| `"A24021 C"` | `"A24021C"` |
| `"core black/core white"` | `"COREBLACK/COREWHITE"` |
| `"WHSL  $65.00"` | `"WHSL$65.00"` |
| `None` | `""` |
| `"\n\t  "` | `""` |

A field is "present" in the page if `canon(field) in canon(text_layer)`.

This is intentionally a substring test, not a tokenizer. Vendor PDFs put
SKUs in tables, captions, and footnotes; tokenizing risks losing them.
Substring containment after canonicalization handles `A24021 C` vs
`A24021C` (inserted/missing whitespace), `BLACK` vs `Black` (case), and
multi-line wrapping (newlines collapsed).

## Per-field policy

The whole policy is in [verify.py:30-150](verify.py). Three tiers:

| Field | Policy if missing from text layer | Why |
|---|---|---|
| **`sku`** | **Drop the entire row.** Log to `rejected_candidates` with `stage: "deterministic"`, `reason: "sku_not_in_text_layer"`. | A row without a real SKU is a hallucination — there's no salvage path. |
| **`cost`** | Clear value to `None`. Set `verification.cost = "not_in_text_layer"`. | Cost is downstream-load-bearing; better to write nothing than write a hallucinated price. |
| **`retail`** | Same as cost. | Same reasoning. |
| **`description`** | Keep the value as-is. Set `verification.description = "unverified"`. | Descriptions often get legitimately normalized by the VLM (curly quotes → straight, en-dash → hyphen). Surfacing the mismatch in REVIEW is more useful than dropping the row. |
| **`color`** | Same as description. | Same: VLM might normalize "Brt Cactus" → "Bright Cactus" — still useful to a human even if unverified. |
| **`intro_date`** | Same as description. | Date strings get reformatted (5/1/25 ↔ MAY 2025) — by design unverified, not dropped. |
| **`gender_hint`** | Skipped entirely. | Sourced from `prev_context.current_section`, not from the page itself. |

The pattern: **cost/retail get cleared because empty is safer than wrong; everything else gets a flag because the value is still useful even unverified.**

## The failure matrix — three ways text-layer verification can't run

All three return `text_layer_present=False` and the caller routes the
candidates verbatim through today's Stage 2 LLM validator unchanged.
Stage 2 is preserved as a safety net, just demoted from "always-on" to
"fallback."

| Case | Cause | `text_layer_error` | What runs |
|---|---|---|---|
| 1. `get_text()` raises | corrupt PDF, encrypted section, OOM on huge pages, PyMuPDF version quirk | `"pymupdf_failed: <Type>: <msg>"` | Stage 2 LLM |
| 2. `get_text()` returns `""` | scanned / image-only page (common in lookbook spreads) | `None` | Stage 2 LLM |
| 3. `get_text()` returns OCR garbage | rare: embedded text that doesn't match anything visible | `None` (looks like case 2 from outside) | Deterministic check runs, rejects every candidate → page ends up with 0 products → existing REVIEW predicate flags it |

For (3), the deterministic check still fires but rejects everything as
`sku_not_in_text_layer`. The page becomes "label=product, 0 products"
which the REVIEW sheet already flags. Soft failure, not a hard error.

## Where it slots in

The splice is in
[`extract_products.process_page`](../extract_products.py) immediately
after Stage 1:

```
candidates, extract_usage = extract_page(client, model, png, prev_context)
record["n_candidates"] = len(candidates)
record["usage"]["extract"] = extract_usage

kept, dropped, text_layer_present, text_layer_error = (
    verify_against_text_layer(candidates, pdf, page_no, _pdf_lock)
)
record["text_layer_present"] = text_layer_present

if text_layer_present:
    # deterministic path: kept + dropped are final, no Stage 2 call
    products  = kept
    rejected  = dropped  # already tagged stage: "deterministic"
    record["usage"]["validate"] = None
else:
    # fallback: today's LLM validator on the original candidates
    products, rejected, validate_usage = validate_skus(client, candidates)
    for r in rejected: r["stage"] = "llm_validator"
    record["usage"]["validate"] = validate_usage
```

Thread safety: `get_text()` shares the same `_pdf_lock` the renderer
already uses ([extract_products.py:274](../extract_products.py#L274)).
PyMuPDF documents are not thread-safe; the lock serializes both
renderers and verify accessors over the same `fitz.Document`.

## Cost and latency

- **LLM cost: $0.** Pure local compute.
- **Wall time per page: 5–50 ms** for `get_text()` + <1 ms for canon +
  substring checks. Cumulatively invisible against Opus's 1–3 s per
  page.
- **Cost savings:** one Sonnet validator call (~$0.005) is skipped per
  text-layer-present page. On Converse this added up to $0.045 saved
  (9 product pages, all with text layer except 2 short-circuited
  empty-SKU pages → 0 Sonnet calls fired).
- **Time savings:** the eliminated Sonnet call had 1–3 s round-trip; on
  Converse extract dropped 59.4 s → 29.6 s (most of that is the round
  trips coming out, parallelized).

## What the verifier reports back

Returns a 4-tuple:

```python
(kept, dropped, text_layer_present, text_layer_error)
```

- `kept` — list of dicts, each is the original candidate with one new
  key: `verification = {sku, description, color, cost, retail, intro_date}`
  where each value is `"ok"`, `"not_in_text_layer"`, or `"unverified"`.
  For `cost`/`retail` marked `not_in_text_layer`, the field itself is
  set to `None`.
- `dropped` — list of original candidate dicts plus
  `reason: "sku_not_in_text_layer"` and `stage: "deterministic"`.
- `text_layer_present` — `True` if the deterministic check ran;
  `False` if the caller should fall back.
- `text_layer_error` — `None` unless PyMuPDF raised, in which case the
  caller stores it on the page JSON for debugging.

## Verified behavior (from injection tests)

Synthetic candidates against the real Nike PDF text layer:

| input | outcome | verification flags |
|---|---|---|
| `sku="JA1013-010"` (real SKU on page 1) | KEPT | all `"ok"` |
| `sku="ZZZZ9999"` (fake) | DROPPED | `stage="deterministic"`, `reason="sku_not_in_text_layer"` |
| `sku="JA1013-100"` (real), `cost="$9999"` (fake price) | KEPT, cost cleared to `None` | `verification.cost = "not_in_text_layer"` |
| `sku="IZ4702-235"` (real), `description="TOTALLY MADE UP NAME"` | KEPT, description preserved | `verification.description = "unverified"` |

Fallback paths (stubbed `fitz.Document`):

| input | outcome |
|---|---|
| `get_text()` returns `""` | `text_layer_present=False`, candidates passed through verbatim |
| `get_text()` returns whitespace-only | `text_layer_present=False` |
| `get_text()` raises `RuntimeError("simulated corrupt page")` | `text_layer_present=False`, `text_layer_error="pymupdf_failed: RuntimeError: simulated corrupt page"` |

## Explicit non-goals

- **Field/SKU misalignment.** If Opus swaps which `cost` goes with which
  SKU on the same page, both costs still substring-match the text layer
  and pass. Catching that needs bbox spatial info — out of scope.
- **Normalization false positives.** `BLK` → `BLACK` or `'07` (straight
  quote) vs `'07` (curly quote) will be flagged as `unverified` even
  though the value is correct. By design — REVIEW exists to surface,
  not block. On both Nike and Converse the only unverified flags came
  from this class of issue, never from real hallucinations.
- **Recall improvement.** Verify cannot add a SKU Opus missed. The
  precision-floor here is bounded above by Stage 1's recall.
- **Confidence beyond binary.** Either the canon'd string is in the
  canon'd text or it isn't. No fuzzy matching, no levenshtein, no
  threshold. Trade-off: very predictable behavior, occasionally too
  strict on punctuation normalization.

## Companion docs

- [../analysis/fill_cells.md](../analysis/fill_cells.md) — how the
  deterministic check's drops/flags surface in the per-PDF report.
- [../analysis/cost.md](../analysis/cost.md) — why this module makes
  `validate` cost go to $0 on most PDFs.
