# kithbuysheet — system overview

## 1. What the system is, in one paragraph

A pipeline that turns a **vendor wholesale shoe catalog (PDF)** into a **filled BUYSHEET xlsx**. It does one Opus VLM call per product page (the expensive part), uses PyMuPDF's text layer to verify the VLM's claims for free, and one Sonnet call per unique dropdown vocab key (cached on disk). Optionally, it also crops each shoe's photo and embeds the thumbnail next to its SKU row in column A. The optional image step uses YOLO-World as the detector and Sonnet to (a) pick how aggressively to search and (b) decide which crop belongs to which SKU.

---

## 2. Mental model: the system has 4 layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  CLI layer        run_pipeline.py  →  worker.run_multi               │
├──────────────────────────────────────────────────────────────────────┤
│  Worker layer     normalize-one-doc → merge-N-docs → build-xlsx      │
├──────────────────────────────────────────────────────────────────────┤
│  Per-doc steps    classify → extract (+ optional detect+match)       │
├──────────────────────────────────────────────────────────────────────┤
│  Building blocks  _render, vocab_map, deterministic_check, analysis, │
│                   detect (optional)                                  │
└──────────────────────────────────────────────────────────────────────┘
```

The trick to reading the codebase: **everything funnels into one shape — a per-page JSON.** Every step reads/writes those JSONs. Excel adapters fake a single-page JSON. The merger reads JSONs from many docs and produces one merged row list. The builder is downstream of everything and only knows about row dicts.

---

## 3. The end-to-end flow of ONE PDF

```
        my_catalog.pdf
              │
              ▼
   ┌──────────────────────┐
   │ _render.py           │  300dpi PNG per page, byte-cap 3.6MB,
   │                      │  adaptive PNG→JPEG fallback for photo-heavy
   └──────────┬───────────┘  decks. Cached in <stem>.pages/NN.png|jpg
              │
              ▼
   ┌──────────────────────┐
   │ classify_pages.py    │  Sonnet vision, per page →
   │                      │   label = brand_name | category |
   │                      │           product | index_or_other | unknown
   │                      │  + carries running {vendor, current_section}
   └──────────┬───────────┘  into the next page's prev_context
              │  writes NN.json
              ▼
   ┌──────────────────────┐
   │ extract_products.py  │  for each label=product page:
   │                      │
   │   ① Opus VLM extract │   one image+context call → 7-field rows
   │      (recall)        │   (sku/desc/color/cost/retail/intro/gender)
   │                      │
   │   ② Verifier         │   deterministic_check.verify_against_text_
   │      (text layer)    │   layer — per-field substring match against
   │                      │   PyMuPDF's text. SKU misses drop; price
   │                      │   misses null; desc/color/intro misses mark
   │                      │   "unverified".  Replaces the old Stage-2 LLM
   │                      │   when text layer is present.
   │                      │
   │   ③ Sonnet fallback  │   only when text layer missing/garbage —
   │      (precision)     │   text-only SKU validator drops obvious
   │                      │   non-SKUs.
   │                      │
   │   ④ Optional image   │   single-doc only — see §6
   │      pipeline        │
   └──────────┬───────────┘  appends products + verification + usage
              │  to NN.json
              ▼
   ┌──────────────────────┐
   │ worker/merge.py      │  N=1: passthrough (single doc trivially
   │                      │  becomes the merged set).
   │                      │  N≥2: see §7 (in review)
   └──────────┬───────────┘  emits products_rows + sku_conflicts
              │
              ▼
   ┌──────────────────────┐
   │ vocab_map.py         │  for each unique (field, key) in rows,
   │                      │  Sonnet picks the dropdown canonical
   │                      │  (or returns null). Cached at cache/<f>.json
   │                      │  — cross-vendor cache hits are real
   │                      │  ("STAN SMITH" survives across docs).
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │ build_buysheet.py    │  load BUYSHEET_template.xlsx, write each
   │ or                   │  product row into TEMPLATE sheet, flag
   │ worker/build_merged  │  problem pages yellow + REVIEW sheet.
   └──────────┬───────────┘  Optionally embed image thumbnails in col A.
              │
              ▼
       BUYSHEET_<vendor>.xlsx
```

---

## 4. The two LLM personas

The pipeline never asks one model to do everything. Each call is shaped to play to a model's strengths.

| Persona | Model | Job | Why |
|---|---|---|---|
| **Reader** | Opus 4.7 (vision) | Extract products from one page image, including off-pattern SKUs | High recall, expensive — runs once per page max |
| **Judge** | Sonnet 4.6 (text + vision) | Classify pages, validate SKUs, pick dropdown vocabs, pick YOLO density, match crops to SKUs | Cheap, fast, easy to cache — runs many times |
| **Verifier** | — | Substring-match VLM output against PDF text layer | Free, replaces a previous LLM call on the deterministic path |

Cost shape on a typical 30-page catalog: **~$3-5 total**, dominated by Opus extract calls.

---

## 5. Storage layout (where things live on disk)

```
my_catalog.pdf
my_catalog.pages/                       ← created by classify, written by every step
  01.png  01.json                       ← classify wrote 01.json; extract appended products
  02.png  02.json
  ...
  _build_usage.json                     ← vocab_map token accounting

my_catalog.pages/07.crops/              ← optional, only if detection ran
  manifest.json                         ← bboxes, crop paths, chosen imgsz + reasoning
  annotated.png                         ← page image with numbered red boxes (matcher input)
  01.png  02.png  03.png  ...           ← per-shoe crops, sorted in reading order

cache/MG.json   cache/SG.json   ...     ← persistent vocab_map cache, cross-doc
analysis/<doc>.json                     ← fill-rate + spend report (auto-runs post-build)
models/yolov8s-worldv2.pt               ← optional YOLO weights, gitignored
```

**Two important invariants:**

1. Every step is **idempotent + resumable**. A crash mid-extract leaves prior page JSONs intact; reruns skip what's done. `--force` re-runs.
2. **JSONs are the source of truth.** Nothing recomputes from cached LLM responses — the JSON IS the cached response.

---

## 6. The optional image pipeline (column A thumbnails) — single-doc only

Activated automatically on single-doc runs when `ultralytics` + a YOLO checkpoint are present. Skipped silently otherwise; the rest of the pipeline is unaffected.

```
page image (already rendered for extract)
        │
        ▼
   ┌────────────────────┐
   │ ① Density picker   │   Sonnet sees a clean page → returns one of
   │    (Sonnet vision) │   {low, medium, high}, mapped to YOLO imgsz
   │                    │   {1280, 1920, 2560}. Bias: when borderline,
   │                    │   pick HIGHER — over-detect is recoverable.
   └─────────┬──────────┘
             │ imgsz cached in manifest.json (no re-pick on rerun)
             ▼
   ┌────────────────────┐
   │ ② YOLO-World       │   stock COCO weights + set_classes(["shoe",
   │    detect          │   "sneaker","boot","sandal"]) at load time.
   │                    │   agnostic_nms=True to merge duplicate class
   │                    │   predictions on the same shoe.
   │                    │
   │                    │   Output: N bboxes, sorted in row-banded
   │                    │   reading order. Crops to NN.crops/MM.png.
   │                    │   Numbered red boxes drawn → annotated.png.
   │                    │
   │                    │   SERIAL — ultralytics.predict() is not
   │                    │   thread-safe.
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐
   │ ③ Matcher          │   Sonnet sees annotated page + extracted SKU
   │    (Sonnet vision) │   list. Returns SKU→box-number map. Hero /
   │                    │   lifestyle / marketing shots return null.
   │                    │   Duplicate-box assignment → first-SKU-wins.
   └─────────┬──────────┘
             ▼
   image_path = NN.crops/<assigned-box>.png   on each product dict
             │
             ▼
   build_buysheet._embed_image  →  OneCellAnchor PNG in cell A<row>
                                   row height bumped to 100pt, col A
                                   widened to ~140 px, "PHOTO" header
                                   preserved from the template.
```

**Why this design, not the simpler "VLM returns bboxes":** VLMs are bad at coordinates but great at semantic matching. So coords come from YOLO, identity-matching comes from Sonnet — each tool plays to its strengths.

**What gets flagged:** if the matcher finds no usable assignment for any SKU on a page, the page record gets `image_association: "no_match"` and shows up in the REVIEW sheet alongside the existing flagging causes (text-layer absent, 0 products, verification issues).

**Measured coverage on real catalogs:**

| | image cov | density mix |
|---|---|---|
| Converse | 15/15 (100%) | 8 low / 1 med |
| Salomon | 147/150 (98.0%) | 29 low / 7 med / 1 high |
| adidas | 314/333 (94.3%) | 6 low / 9 med / 9 high |

---

## 7. ⚠️ Multi-doc support — still in review

The worker can take multiple input docs and produce one merged buy-sheet. It's wired end-to-end (`worker/run_multi.py`, `worker/merge.py`, `worker/build_merged.py`) but **has not been tested with the depth Converse / adidas / Salomon got.** Known caveats:

| Aspect | Status |
|---|---|
| SKU canonicalization (whitespace+case-stripped) for grouping across docs | implemented, untested at scale |
| First-non-empty-wins per field, CLI-order priority | implemented |
| Conflict policy (keep rows separate, SKU_CONFLICTS sheet) when sources disagree on a non-empty field | implemented, **edge cases not yet exercised** — multi-doc disagreements have only been validated on synthetic test cases |
| Single-vendor strict mode (every doc must declare a vendor; detected vendors must agree) | implemented |
| Excel adapter (no LLM, header-synonym match) feeds into the same JSON shape | implemented, lightly tested |
| **Image column on multi-doc** | **explicitly disabled** — detection only runs when `len(docs) == 1`. Cross-doc image merging (which crop wins per SKU) is out of scope until the matcher's semantics are extended to handle it |
| Auto-priority advisor (one cheap probe per doc → priority ranking + per-field overrides) | designed in [worker.md](worker.md), **not implemented** |
| PPTX adapter | not implemented — convert externally first |

**Translation:** single-doc runs are production-quality. Multi-doc is a working prototype where the merge logic needs more empirical validation before you'd trust it on real vendor disagreement cases.

---

## 8. Caching strategy (why re-runs are fast and cheap)

Three independent caches:

| Cache | Where | What it skips on rerun | Invalidate with |
|---|---|---|---|
| **Page renders** | `<doc>.pages/NN.png` | PyMuPDF rasterization | Delete the file |
| **Per-page JSON** | `<doc>.pages/NN.json` | Whole step (classify *or* extract+verify+match) | `--force` flag on the step's CLI |
| **Vocab dropdown** | `cache/<field>.json` (cross-doc, cross-vendor) | Sonnet vocab_map calls | Delete the cache file |
| **YOLO detection** | `<doc>.pages/NN.crops/manifest.json` | Picker + detector + crop save | `--force` (or delete the dir) |

The vocab cache is the unsung hero: adidas → Converse → Salomon runs share entries like `"STAN SMITH"` or `"core black"`. By the third catalog, vocab_map cost is near zero.

---

## 9. Failure surfacing (how problems become visible)

You don't need to read every JSON to find issues:

- **Yellow STYLE# tint** on any row whose source page got flagged (extraction error, 0 products on a `product` page, missing text layer, any unverified field, or matcher couldn't place an image)
- **REVIEW sheet** appended to every output xlsx — one row per flagged page with `verification_issues` summary
- **SKU_CONFLICTS sheet** (multi-doc only) — when two sources disagree on a SKU's non-empty fields, both rows kept + listed here
- **`analysis/<doc>.json`** + stderr summary — fill-rate per field, SKU drop counts by stage, $ spend per LLM step

---

## 10. Tweak surfaces (things you'll actually want to change)

| What | Where |
|---|---|
| Classify prompt | `CLASSIFY_SYSTEM` in [classify_pages.py](classify_pages.py) |
| Extract prompt | `EXTRACT_SYSTEM` in [extract_products.py](extract_products.py) |
| Density picker prompt | `YOLO_SETTINGS_SYSTEM` + `_DENSITY_TO_IMGSZ` in [extract_products.py](extract_products.py) |
| Matcher prompt | `IMAGE_MATCH_SYSTEM` in [extract_products.py](extract_products.py) |
| Vocab-map prompt | `SYSTEM` in [vocab_map.py](vocab_map.py) |
| Excel column synonyms | `SYNONYMS` in [worker/formats/excel_adapter.py](worker/formats/excel_adapter.py) |
| YOLO weights | `models/yolov8s-worldv2.pt` or `$DETECT_WEIGHTS` env var |
| Image cell size | `IMAGE_TARGET_PX` / `IMAGE_ROW_HEIGHT_PT` / `IMAGE_COL_WIDTH` in [build_buysheet.py](build_buysheet.py) |
| Workers | `--workers N` on every CLI |

---

## 11. Key files at a glance

| File | One-line role |
|---|---|
| [run_pipeline.py](run_pipeline.py) | User-facing CLI shim → `worker.run_multi.main` |
| [worker/run_multi.py](worker/run_multi.py) | Single CLI entry; classify+extract per doc, then merge+build |
| [worker/formats/pdf_adapter.py](worker/formats/pdf_adapter.py) | Per-doc dispatch into classify+extract |
| [worker/formats/excel_adapter.py](worker/formats/excel_adapter.py) | Deterministic Excel → synthetic page JSON |
| [worker/merge.py](worker/merge.py) | SKU-grouped merge with conflict detection (multi-doc) |
| [worker/build_merged.py](worker/build_merged.py) | xlsx writer for merged rows, REVIEW + SKU_CONFLICTS sheets |
| [classify_pages.py](classify_pages.py) | Step 1: Sonnet page-type classifier |
| [extract_products.py](extract_products.py) | Step 2: Opus extract + verifier + Sonnet validator + density picker + matcher |
| [_render.py](_render.py) | Adaptive PDF→PNG/JPEG renderer with byte/dim caps |
| [deterministic_check/verify.py](deterministic_check/verify.py) | Text-layer substring verifier |
| [vocab_map.py](vocab_map.py) | Sonnet dropdown vocabulary mapper with on-disk cache |
| [build_buysheet.py](build_buysheet.py) | Step 3: writes BUYSHEET xlsx, including optional image embed |
| [detect/shoe.py](detect/shoe.py) | Optional footwear detection (YOLO-World loader, crops, annotation) |
| [analysis/fill_rate.py](analysis/fill_rate.py) | Post-run report, no LLM |
| [analysis/usage.py](analysis/usage.py) | Token + cost accounting |

---

## 12. One-page cheat sheet

- **Run one PDF:** `python run_pipeline.py <doc.pdf>`
- **Run with images:** same command, plus `pip install -r requirements-detect.txt` once, and weights in `models/yolov8s-worldv2.pt`
- **Re-run with fresh extraction:** add `--force`
- **Single-doc = production. Multi-doc = beta** (works, but the conflict-resolution path needs more real-world testing before you trust it on vendor disagreements)
- **All state is on disk in `<doc>.pages/`** — delete it to wipe one doc's cache; delete `cache/` to wipe vocab lookups
- **Spend tracker is at `analysis/<doc>.json`** and printed at end of every run
