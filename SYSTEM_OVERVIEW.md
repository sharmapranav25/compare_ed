# Photo matching — system overview

This branch (`photo_matching_yolo`) adds **Step 3: photo localization** to
the existing buy-sheet pipeline. After `classify_pages.py` + `extract_products.py`
have produced per-page SKUs, this step figures out *which rectangle on
the page is the product photo for each SKU* and writes the bbox + a
cropped PNG back into the page JSON.

The output is consumed downstream by the buy-sheet builder (for embedding
the right photo next to the right STYLE#) and by any reviewer UI that
wants to render "SKU → photo" side-by-sides.

## Where this fits

```
┌──────────────────────────┐   ┌──────────────────────────┐   ┌───────────────────────┐
│ Step 1 classify_pages.py │ → │ Step 2 extract_products  │ → │ Step 3 match_photos   │
│ Sonnet VLM, one call /pg │   │ Opus VLM + Sonnet text   │   │ YOLO + PyMuPDF + geom │
└──────────────────────────┘   └──────────────────────────┘   └───────────────────────┘
            │                              │                              │
            ▼                              ▼                              ▼
     <NN>.json                     <NN>.json adds:               <NN>.json adds:
     label, vendor,                  products[],                    products[].photo_box,
     prev_context                    rejected_candidates            photo_crop, match_confidence
                                                                   <NN>.match.json (full log)
                                                                   <NN>.audit.png (overlay)
                                                                   _audit_index.html
```

Step 3 is independent of the VLM steps — given an existing
`<doc>.pages/` directory with extractions, it can re-run in isolation
(`python match_photos.py <pdf>`).

## Inputs and outputs

| Per-page input | Written by | What it contains |
|---|---|---|
| `<NN>.json` | classify + extract | `products[]`, `prev_context.vendor` |
| `<NN>.yolo.json` | extract (sidecar) | `boxes: [{cls, bbox, conf}]` from DocLayout-YOLO |
| `<NN>.pymupdf.json` | extract (sidecar) | `words[]` + `image_rects[]` from PyMuPDF |
| `<NN>.fullres.png` | extract (sidecar) | 300dpi render the boxes are in pixel space of |

| Per-page output | Purpose |
|---|---|
| `<NN>.match.json` | Full decision log — every candidate, every filter decision, all three matching streams (yolo-only, pymupdf-only, combined), agreement / confidence labels |
| `<NN>_<sku>.png` | Cropped product photo, one per matched SKU |
| `<NN>.audit.png` | Pillow overlay on the fullres page — kept/dropped candidates, SKU-word bboxes, SKU→photo lines color-coded by agreement |
| `_calibration.json` | Doc-wide blob: per-vendor size/aspect priors, per-vendor distance + direction priors, recurring-xref count |
| `_audit_index.html` | Browseable index linking every audit PNG with per-page stats |

The per-product fields added back to `<NN>.json`:

```json
{
  "sku": "A24021 C",
  "description": "JACK PURCELL",
  "...": "...",
  "photo_box": {"x": 412.0, "y": 188.0, "w": 240.0, "h": 240.0},
  "photo_box_source": "both",
  "match_confidence": "high",
  "photo_crop": "07_A24021_C.png"
}
```

## Why two detection streams

A wholesale buy-sheet PDF is one of three things:

1. **Native InDesign / typesetting output** — every image is a discrete
   PDF XObject. PyMuPDF enumerates them exactly, no inference.
2. **Rasterized scan / flattened export** — the page is one big image.
   PyMuPDF sees nothing useful; we need a vision model.
3. **Mixed** — text is selectable but the photos are baked into one
   flattened raster (common when a vendor "saves as PDF" from Keynote).

Running both detectors in parallel and reconciling them at match time:

- **PyMuPDF** is exact when it works. Cheap. But returns zero hits on
  rasterized pages.
- **DocLayout-YOLO** (DocStructBench weights) is heuristic but always
  produces something. On rasterized pages it's the only signal.
- **Agreement (IoU ≥ 0.5) ⇒ "both" with high confidence.** Disagreement
  forces a tie-break by per-vendor distance calibration.

`extract_products.py` runs both detectors during Step 2 and writes the
sidecar JSONs; `match_photos.py` consumes them in Step 3.

## The matching algorithm

```
1. Aggregate yolo + pymupdf observations across every product page.
2. Compute doc-wide xref page-counts → flag recurring decorative xrefs
   (vendor logos, page-frame chrome) as "kill these candidates."
3. Compute per-vendor YOLO figure-box size/aspect priors. Independently
   compute PyMuPDF size priors (YOLO sometimes collapses a dense
   product grid into one huge figure box — using its median to filter
   PyMuPDF would reject every real photo).
4. Pass 1 — naive nearest-neighbor on size/aspect-filtered candidates.
   Keep ONLY pairings where YOLO and PyMuPDF agreed (IoU ≥ 0.5). These
   become the bootstrap set.
5. From the agreed pairings, calibrate per-vendor SKU→photo distance
   (median + IQR-derived σ) and direction (circular mean + concentration).
   This learns "for ADIDAS the photo sits ~180px above and slightly
   right of the SKU" from the page itself, no hand-tuned priors.
6. Pass 2 — re-match every SKU against the filtered candidate list using
   a soft score:
       score = ((distance − expected_d) / σ)²
             + DIRECTION_WEIGHT · (1 − cos(angle − expected_a))
   Three parallel streams: yolo-only, pymupdf-only, combined.
7. Persist <NN>.match.json + update <NN>.json product fields. Crop each
   matched photo to <NN>_<sku>.png.
8. Render audit PNGs + a browseable HTML index for human review.
```

## Confidence labels

A combined-match score gets bucketed:

| Score | Label | Meaning |
|---|---|---|
| ≤ 2.0 | `high` | Tight on both distance and direction priors |
| ≤ 8.0 | `medium` | One axis off; still the best candidate |
| > 8.0 | `low` | Likely wrong — flag for human review |

If YOLO and PyMuPDF disagree (IoU < 0.5) on the chosen box, a `high`
gets downgraded to `medium`. The buy-sheet builder uses this label to
decide whether to embed the photo automatically or route the row to a
review queue.

## Calibration tiers — when each kicks in

For each vendor, the matcher picks the best-supported priors:

- **per-vendor**: vendor has ≥ 5 product pages of YOLO observations.
  Uses that vendor's own size / aspect / distance / direction priors.
- **doc-wide**: small vendors fall back to the document-wide pooled
  priors. Single-vendor catalogs always end up here, which is fine
  because all pages share the same layout.
- **bootstrap (no distance prior)**: if Pass 1 yielded fewer than 5
  agreed pairings, distance calibration is skipped and Pass 2 falls
  back to raw nearest-neighbor on the filtered candidate set.

The tier used per page is recorded in `<NN>.match.json.calibration_tier`.

## File map (this branch only)

| File | Role |
|---|---|
| `match_photos.py` | Step 3 entry point. Implements the 8-step algorithm above. CLI: `python match_photos.py <pdf>` (with `--force`, `--page N`). |
| `_layout.py` | DocLayout-YOLO loader + per-vendor size/aspect calibration. Lazy-loads weights from `juliozhao/DocLayout-YOLO-DocStructBench` into `~/.cache/doclayout-yolo/`. Raises `YoloUnavailable` if `doclayout-yolo` / torch isn't installed (callers degrade gracefully). |
| `_pymupdf_obs.py` | PyMuPDF observation gatherer: word list + image-XObject rects in pixel space of the matching fullres render. Also has `find_sku_word_bbox` — stitches adjacent word bboxes to locate a multi-token SKU on the page. |
| `_overlay.py` | Audit visualization: `render_audit()` draws all candidates / decisions / links on the fullres page; `render_index()` builds a static HTML index. Style constants in one `_STYLE` dict at the top. |
| `probe_layout.py` | Single-page probe — runs the whole flow (or just YOLO) on one PNG or one PDF page. Outputs land under `probes/<doc-stem>/`. Use for prompt iteration and YOLO sanity-checking. |

## Tuning surface

Constants at the top of `match_photos.py`:

- `RECURRING_XREF_FRACTION` (0.6) — xrefs appearing on >60% of pages
  are killed as decorative chrome.
- `SIZE_TOL` / `ASPECT_TOL` (0.6 log₂) — YOLO size band — roughly a
  factor-of-1.5 around the per-vendor median.
- `PYMUPDF_MIN_DIM_PX` (40), `PYMUPDF_MAX_ASPECT` (5.0) — kill swatches,
  color dots, and header banners from the PyMuPDF candidate set.
- `PER_VENDOR_MIN_PAGES` (5), `BOOTSTRAP_MIN_PAIRS` (5) — tier thresholds.
- `AGREE_IOU` (0.5) — IoU threshold for "YOLO and PyMuPDF agreed".
- `DIRECTION_WEIGHT` (0.5), `HIGH_SCORE` / `MEDIUM_SCORE` — confidence
  bucketing.

## Dependencies

- `doclayout-yolo` (+ torch backbone) — heuristic detector. The first
  call downloads weights via `huggingface_hub`.
- `PyMuPDF` (`fitz`) — exact PDF image-rect + word enumeration.
- `Pillow` — fullres rendering + audit overlay.

If `doclayout-yolo` isn't installed, the YOLO stream degrades to empty
and matching runs PyMuPDF-only. The pipeline still completes for native
PDF inputs; rasterized pages get no matches and are flagged for review.

## Running it

```bash
# Full pipeline including photo match:
python classify_pages.py  <pdf>
python extract_products.py <pdf>      # also writes .yolo.json + .pymupdf.json sidecars
python match_photos.py    <pdf>       # this step
python build_buysheet.py  <pdf>

# Single page (debug):
python match_photos.py <pdf> --page 7

# Force re-match an already-matched page:
python match_photos.py <pdf> --force

# YOLO sanity check, no PDF needed:
python probe_layout.py path/to/page.png
```
