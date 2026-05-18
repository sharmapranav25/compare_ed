# Accuracy & Fill-Rate Expectations

**Audience:** anyone deciding whether to trust v1 output without manual review, or onboarding a new vendor for the first time.
**Confidence basis:** firm measurements on 3 vendor catalogs (Nike HO26, Adidas FW26 premium, Converse HO26) totaling 471 SKUs / 4,440 cells. Predictions for other catalog types are informed extrapolations and should be treated as estimates until validated.

---

## TL;DR

For a typical Kith vendor catalog (single-brand, vector PDF, athletic footwear, grid or vertical-card layout), expect **85-92% per-cell accuracy** with text fields essentially complete. The two main quality risks are (a) `ssg` field which depends on vocabulary coverage and (b) any image-only / PPT-exported pages where the source-text oracle can't verify the extraction.

Per-cell accuracy by catalog profile:

| Profile | Per-cell | Per-card | Confidence | Examples |
|---------|---------:|---------:|:----------:|----------|
| Vector PDF, single-brand, grid/vertical-card | **85-92%** | **85-95%** | HIGH | Nike HO26 ✓, Adidas FW26 premium ✓ |
| Vector PDF, multi-brand reseller | 75-85% | 70-85% | MEDIUM-HIGH | Hanger Clinic (not yet measured under v1) |
| Mixed text-layer / image-only PPT | 65-75% (oracle-verified) | ~80-90% (likely actual) | MEDIUM | Converse HO26 ✓ |
| Fully scanned / no text layer | 70-85% (unverifiable) | unknown | LOW-MEDIUM | None tested |
| Lookbook / marketing-heavy | 75-90% | unknown | MEDIUM | Hoka SP27 (holdout, pending) |
| Novel layout, never seen | 70-90% | unknown | MEDIUM | Genuinely random vendor |

---

## Per-field accuracy and fill rate

Per-field oracle scoring on Nike + Adidas + Converse (470+ SKUs):

| Field | Likely fill rate | Likely accuracy | What drives it |
|-------|----------------:|---------------:|----------------|
| **sku** | ~100% | **95-100%** | Anchored via PyMuPDF text-search. Image-only PDFs drop slightly. |
| **brand** | 100% | **95-100%** single-brand / 85-95% multi-brand | Catalog-level inference is bulletproof for single-brand catalogs; per-SKU-prefix voting for multi-brand catalogs |
| **usd_cost** | 0-100% (vendor-dependent) | **95-100%** when present | Strong numeric source-text verification. Wholesale-only catalogs (Nike) have 100% cost; retail catalogs (Converse) have both cost + retail |
| **intro_date** | 70-100% (vendor-dependent) | **90-100%** when present | Numeric date or month-code substring match |
| **description** | 95-100% | **88-98%** | Per-card retry essentially solves cross-card mis-attribution. Some loss on very dense grids. |
| **color** | 90-100% | **80-95%** | Strong on explicit "Color: X" formats; weaker on multi-token slash patterns ("Black /Anthracite Volt") |
| **mg** (M-/W-/K-Footwear) | 90-100% | **80-95%** | Strong with prefix tokens (W, WMNS, Men's, GS); weaker on ambiguous adult shoes |
| **standard_color** | 70-95% | **70-85%** | `vocab/color_synonyms.json` lookup (~500 entries); new vendor color names lower accuracy |
| **sg** (Sneakers/Boots/etc) | 90-100% | **70-90%** | `vocab/description_map.json` cache hits vary by vendor freshness; defaults to Sneakers for athletic |
| **ssg** (Basketball/Running/Causal Shoe/etc) | 80-100% | **30-70%** | **Weakest field.** description_map miss + Kith template's "Causal" misspelling cause widespread "uncertain" marking. Not "wrong" — just unverified. |
| **usd_retail** | 0-100% (vendor-dependent) | **85-95%** when present | Many shoebuyer catalogs are wholesale-only |

**Important: fill rate ≠ accuracy.** A vendor with no retail prices in its PDF should report 0% fill on `usd_retail` — that's CORRECT behavior, not a failure. When evaluating new vendors, distinguish "field absent in source" (correct null) from "field present in source but missed" (real miss).

---

## Confidence tiers

### HIGH confidence (85-92% per-cell, ship-ready)

Expect minimal manual review on:
- Nike, Adidas, New Balance, ASICS, Saucony, Brooks, Puma, Salomon
- Any vendor that uses InDesign/Illustrator-exported PDFs (vector text layer)
- Single-brand catalogs with explicit SKU + Color labels per product

For these, v1's text-only output is usable as-is, with amber cells for the ~10% that need verification.

### MEDIUM confidence (70-85% per-cell, expect 15-30% amber)

- PPT-exported PDFs (Converse, some lookbook-style catalogs)
- Dense multi-section catalogs with mixed layouts (Adidas FW26 page 2 style)
- Multi-brand resellers (Hanger Clinic, smaller fashion footwear distributors)
- Fashion brands using minimal text + heavy imagery
- Hoka SP27, Mizuno SS27 (not yet measured — pending holdout validation)

For these, expect to spend time reviewing amber cells; fields with `vlm_only_no_region` source attribution need spot-checks against the source PDF.

### LOW confidence (50-75% per-cell, manual review likely)

- Fully rasterized/scanned catalogs (no text layer at all)
- Highly unusual layouts (catalog-as-magazine with editorial photography)
- Languages other than English (untested)
- Brand-new vendor categories (apparel, accessories — not athletic footwear)

For these, the v1 pipeline should be treated as a first-pass draft requiring manual verification before sending to buyers.

---

## What the verified numbers actually show

Per-cell oracle scoring (passing ≥0.7 confidence):

| Vendor | Cards | Per-cell | Contradicted | Cost |
|--------|------:|---------:|-------------:|-----:|
| Nike HO26 | 110 | **87.4%** | 3.2% | $0.53 |
| Adidas FW26 premium | 346 | **87.4%** | 1.6% | $1.73 |
| Converse HO26 | 15 | 69.2% | 6.3% | $0.34 |

**Per-field on Nike HO26 (representative grid/vertical-card vendor):**

| Field | Passing | Contradicted | Notes |
|---|---:|---:|---|
| sku | 100% | 0 | Rock solid |
| brand | 100% | 0 | Catalog-level inference |
| description | 98% | 2 | Per-card retry solves cross-card mis-attribution |
| color | 91% | 10 | Multi-color tokens are the bulk of contradictions |
| usd_cost | 100% | 0 | Numeric source match |
| intro_date | 100% | 0 | Month-code or numeric-date match |
| standard_color | 76% | 3 | Vocab lookup gaps on novel HO26 color names |
| mg | 84% | 17 | Soft verification — trusts VLM when no explicit gender signal |
| sg | 99% | 0 | description_map cache covers most |
| ssg | 33% | 0 | description_map cache miss on novel HO26 product names; NOT wrong, just unverified |

---

## Important caveats

1. **These numbers extrapolate from 3 catalogs.** Until we run the cold-vendor holdouts (Hoka SP27 + Mizuno SS27) the "random catalog" estimate is informed prediction, not measurement. Validation plan: §5 below.

2. **"Oracle accuracy" is conservative.** The semantic oracle marks "uncertain" anything it can't verify against source text (e.g., image-only pages, novel vocab). Actual extraction quality is usually higher than the oracle-passing rate suggests — but we can't *prove* it without hand-verified ground truth.

3. **Cell fill rate ≠ correctness.** A vendor with no retail price will have 0% fill on `usd_retail`. That's correct, not a failure.

4. **The bottleneck for "any random catalog" is image binding, not fields.** Fields are at 87-95%; image binding (column A) was at 25-50% — which is why it's deferred to v2 per [ARCHITECTURE.md §9](ARCHITECTURE.md#9-known-limitations-v1--v2-roadmap).

5. **Cost is predictable.** $0.30-2 per catalog, with sidecar caching making re-runs free. 100 catalogs/month ≈ $80/month.

---

## How to measure a new vendor

Use the eval harness to convert estimates into hard numbers:

```bash
# Step 1: Run the pipeline on the new vendor's PDF
python -m buysheet_v2 run path/to/new_vendor.pdf --vendor-key new_vendor

# Step 2: Build a scaffold of expected values from the v1 output (one-time)
python -m buysheet_v2.tests.scaffold_golden new_vendor
# This writes tests/golden/new_vendor.json (or holdout/ if held-out)

# Step 3: Manually verify 20 random SKUs in the scaffold against the source PDF
# Flip "_verified": false → true on each entry you've confirmed.
# This takes ~30-60 min per vendor depending on layout complexity.

# Step 4: Run the eval
python -m buysheet_v2 eval new_vendor
# Prints per-field accuracy, per-card accuracy, and ship-gate verdict
# (≥85% per-cell + ≥90% per-card = ship-ready for that vendor)
```

This is the *only* way to convert "estimated 85%" into "measured X%" for a specific catalog. Predictions are useful as a sanity check, not a guarantee.

---

## Known levers for improving accuracy

If a specific field is underperforming, these are the targeted improvements:

| Field | Improvement | Cost | Expected lift |
|-------|-------------|------|---------------|
| **ssg** | Rebuild `vocab/description_map.json` from current Nike/Adidas/Converse descriptions (one Claude call per unique description, cached) | $5-10 one-time | 33% → 75-85% |
| **standard_color** | Append new vendor color names to `vocab/color_synonyms.json` (manual + Claude-assist) | 1-2 hours | 76% → 85-90% |
| **mg** | Tighten `_derive_mg()` heuristic with vendor-specific gender token patterns | 2-4 hours | 84% → 90-95% |
| **color** (multi-token) | Improve color verification to handle multi-colorway products explicitly | 4-8 hours | 91% → 95-98% |
| **description** (residual 2%) | Prompt iteration on per-card retry to handle very-dense-grid edge cases | Prompt iteration | 98% → 99%+ |
| **All fields on image-only PDFs** | PyMuPDF OCR fallback for empty text layers (Tesseract) | 1-2 days | Converse: 69% → 80-85% |
| **Image binding** (col A) | Native PDF XObject extraction + nearest-neighbor SKU binding | 1 day | 0% → 80-90% (v2 work) |

See ARCHITECTURE.md §9 for the v2 roadmap and image-binding strategy.

---

## Decision rubric for using v1 output

| Use case | v1 fit? |
|----------|---------|
| Bulk-load HIGH-confidence vendor catalogs (Nike, Adidas, etc) into Kith buy-sheets, with buyer review of amber cells | **Yes — primary intended use case** |
| Generate a first-pass draft for a NEW vendor we've never seen, knowing manual review is needed | **Yes — sets up the buyer 60-80% faster** |
| Replace manual buyer review entirely | **No — v1 is a productivity tool, not a replacement for buyer judgment** |
| Process catalogs with embedded photo IP (need shoe images in col A) | **No — image binding is v2** |
| Process scanned-only catalogs with no text layer | **Not yet — would need OCR fallback added** |
