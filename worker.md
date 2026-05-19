# Worker — buy-sheet pipeline (single + multi-doc)

The single source of truth for the buy-sheet pipeline. Takes N vendor
docs (PDF + Excel) and produces one `BUYSHEET_<vendor>.xlsx`. N=1 is
the trivial case; N≥2 merges by canonicalized SKU with priority-based
field resolution.

User-facing CLI is [run_pipeline.py](run_pipeline.py) at the repo root
— a thin shim that calls `worker.run_multi.main()`. Same effect as
invoking the module directly:

```
single doc:   python run_pipeline.py files/converse_HO26.pdf
multiple:     python run_pipeline.py files/nike/pricing.xlsx files/nike/catalog.pdf
equivalent:   python -m worker.run_multi <args>          # internal module path
```

The leftmost doc on the command line is the highest-priority source.
"Priority" here means: when the same SKU appears in two docs and both
have a non-empty value for the same field, the higher-priority doc wins.

---

## Two layers: worker vs scheduler

| Layer | Lives in | Scope | Status |
|---|---|---|---|
| **Worker** (per-job) | this folder | One vendor folder → one xlsx. Parallel workers within each doc; sequential across docs. | Built (milestone 1) |
| **Scheduler** (cross-job) | not built | Many vendors / many users → fairness, queue, rate budget across runs. | Deferred — seams preserved (see below) |

The naming reflects that distinction: this folder is the *worker* logic
the scheduler will eventually dispatch onto a worker pool. Calling it
`scheduler/` would have collided with that future layer.

---

## Folder layout

```
worker/
  __init__.py                empty
  run_multi.py               CLI entry point + orchestration
  merge.py                   group by SKU, first-non-empty-wins per field
  build_merged.py            write merged xlsx (reuses build_buysheet helpers)
  formats/
    __init__.py              empty
    pdf_adapter.py           thin wrapper over classify_pdf + extract_pdf
    excel_adapter.py         deterministic openpyxl reader, no LLM
```

7 files total, ~400 LOC. Nothing else in the repo was touched.

---

## End-to-end algorithm

```
1. Normalize each input doc to the same on-disk shape
   for each doc in CLI order:
     PDF   → pdf_adapter.normalize  → delegates to existing classify+extract
     Excel → excel_adapter.normalize → openpyxl + synonym table → 00.json
   output of this phase: <doc>.pages/*.json (existing per-page shape)

2. Merge across docs (no API calls)
   read every doc's .pages/, tag products with (priority, source),
   group by canonicalized SKU, per-field-first-non-empty-wins.
   Produces products_rows in the exact shape build_buysheet expects.

3. Build merged xlsx
   reuse gather_unique_tasks + resolve_all + write_rows from
   build_buysheet (imported, not duplicated). vocab_map cache shared
   with single-doc runs, so cross-doc cache hits are free.

4. REVIEW sheet (local to build_merged)
   union of per-doc page records, with a `source` column so a flagged
   page tells you which input doc owns the problem.
```

---

## Per-format adapters

### PDF
[worker/formats/pdf_adapter.py](worker/formats/pdf_adapter.py) is ~10
lines. It calls existing `classify_pdf` then `extract_pdf` and returns
the `.pages/` path. **All PDF behavior — VLM calls, the deterministic
text-layer verifier, Stage-2 SKU validator, caching, parallelism — is
inherited from the single-doc pipeline unchanged.**

### Excel
[worker/formats/excel_adapter.py](worker/formats/excel_adapter.py) is
deterministic-first per the design rule "only spend tokens on judgment
calls." A static synonym table maps headers like `Style #`, `MSRP`,
`Wholesale`, `Color Name` to canonical product fields. Header detection
scans the first 20 rows for the row with the most synonym hits (and
both required fields `sku` + `description` mapped). Data rows below it
become products. Vendor is inferred from the filename slug
(`nike_pricing.xlsx` → `NIKE`); override via `--vendor`.

Every Excel-derived product is tagged `verification: "deterministic"`
on each field — a new value alongside the existing `"ok"`,
`"unverified"`, `"not_in_text_layer"` from the PDF text-layer verifier.
This tells downstream audit code "trusted because openpyxl read it,
not because the VLM matched the PDF text layer."

### PPTX (deferred)
PPTX → PDF via headless LibreOffice → existing PDF path. Not built in
milestone 1. The user noted Nike's deck has already been converted
externally; the pipeline accepts the converted PDF as-is.

---

## Merge policy

Priority = position on the command line (index 0 = highest).

For each canonicalized SKU group, the merge first checks **whether any
field disagrees across the group** (after value normalization — see
below). Two branches from there:

### Branch A — no conflicts → collapse to one row
For each field in `(sku, description, color, cost, retail, intro_date,
gender_hint)`, walk the group sorted by priority ascending and take the
first non-empty value. Verification markers merge per-field too: each
marker comes from whichever source supplied the winning value for that
field. This handles the typical multi-doc case (Excel pricing fills in
cost/retail, PDF catalog fills in description/color).

### Branch B — any conflict → keep ALL occurrences as separate rows
If any non-empty field has ≥2 mutually-non-equivalent values across the
group, **every entry in the group is written as its own row in the xlsx
with the STYLE# cell tinted yellow**. The merge refuses to silently
pick one over the other; the reviewer sees both (or all N) values
side-by-side and resolves by hand. The conflict detail (which SKU,
which fields disagreed, which value each source supplied) lives in a
new `SKU_CONFLICTS` sheet alongside `REVIEW`.

### Value equivalence rules

- **cost, retail** — parsed to numbers via `build_buysheet.parse_money`
  and compared rounded to 2 decimals. So `"WHSL $120"`, `"MSRP $120"`,
  `"120.0 USD"` are all equivalent to `120.0`. If either side fails to
  parse, falls back to text comparison.
- **description, color, intro_date, gender_hint** — whitespace
  collapsed, stripped, lowercased. So `"HANDBALL SPEZIAL"`,
  `"handball  spezial"`, `" Handball Spezial "` are all equivalent.
- **sku** — never considered conflicting; by construction every entry
  in the group has the same canon SKU.

### Real example (adidas Premium Range FW26, page 8 + page 11)
The catalog lists the same canonical SKU `KJ9968` (Handball Spezial) on
both pages — but with different retail ($120 vs $110), intro_date
(Aug vs Jul), and gender_hint (MENS vs WOMENS). Before this rule the
merge silently picked the page-8 values. After this rule, both rows
appear in the TEMPLATE sheet with yellow STYLE# fill, and a
`SKU_CONFLICTS` entry shows exactly which fields disagree and what each
page said — so the reviewer can decide whether the catalog is wrong or
it's actually two distinct line items sharing a code.

Context (`vendor`, `current_section`) on the collapse path comes from
the highest-priority entry in the group, since it drives `vocab_map`
cache keys for MG / SG / SSG. On the conflict path, each kept row
carries its own original context (so MG resolution still gets the right
gender_hint per occurrence).

Per-field manifest overrides (a doc explicitly winning specific fields
regardless of position) are deferred — the basic priority order
subsumes most real cases.

---

## Integration with existing folders

### [deterministic_check/](deterministic_check/)
Inherits automatically through `pdf_adapter` (which calls
`extract_pdf`, which invokes `verify_against_text_layer` per page).
Excel skips it — there's no parallel "text layer" for an openpyxl cell
to disagree with, the cell IS the source. Excel adapter writes
`verification: "deterministic"` instead. **`deterministic_check/verify.py`
itself was not touched.**

### [analysis/](analysis/)
The per-doc `analyze()` in [analysis/fill_rate.py](analysis/fill_rate.py)
is called once per normalized doc, same as in `run_pipeline.py`. A
merged-level analyzer (`analysis/merged_fill_rate.py`) that reports
cross-doc fill rate and per-source attribution is deferred to a later
milestone. **`analysis/fill_rate.py` itself was not touched.**

---

## Execution: sequential across docs, parallel within

```
across docs:   doc1 ─► doc2 ─► doc3        (sequential, default)
within a doc:  page1 ║ page2 ║ page3 …     (parallel, --workers N, as today)
```

Why sequential across docs:

1. **Vocab cache warms between docs.** Doc 1's mapping of `"BLUSHED
   STONE / SAIL"` to a standard color lands on disk before Doc 2
   starts. Parallel docs would race the LLM for the same value.
2. **Rate-limit headroom.** N workers in-flight, not N × docs.
3. **Readable logs, clean failure isolation.** Crash on Doc 2 page 47
   is unambiguous; parallel docs interleave stderr.
4. **Memory.** One PDF's PNGs at a time, not all of them simultaneously.
5. **Resumability already pays the latency back.** `.pages/*.json`
   survive a crash; reruns skip cached pages.

A `--parallel-docs` flag (deferred) would opt into across-doc
parallelism for the warm-cache / many-small-docs case.

---

## CLI

```bash
# Two docs, leftmost wins on field conflicts
python -m worker.run_multi files/converse_HO26.pdf files/nike_HO26.pdf

# Override vendor (also affects Excel filename inference)
python -m worker.run_multi a.pdf b.xlsx --vendor NIKE

# More workers, custom output path, sonnet for extract
python -m worker.run_multi a.pdf b.xlsx --workers 8 --model sonnet --out out.xlsx

# Force re-normalize even if .pages/ already cached
python -m worker.run_multi a.pdf b.xlsx --force
```

Flags mirror `run_pipeline.py`. The single-doc CLI keeps working
unchanged; nothing here replaces it.

---

## Multi-vendor input

Milestone 1 enforces single-vendor strictly. The run aborts when:
- **any** doc has no detectable vendor (PDF lacks a `brand_name` page,
  Excel filename has no recognizable prefix), OR
- detected vendors disagree across docs, OR
- no doc declared a vendor at all.

Reason: silently merging Nike's `AB1234` with Adidas's `AB1234`
(unrelated SKUs that happen to collide) would corrupt the output.
Earlier the guard only fired on case 2 (disagreement among *detected*
vendors), which silently passed when one doc happened to lack a brand
cover — exactly Nike's catalog shape. Strict mode closes that gap.

Two ways out:
1. Pass `--vendor <NAME>` to force a single label and proceed; the
   user takes explicit ownership.
2. Run each vendor's docs as a separate `worker.run_multi` invocation.

Automatic multi-vendor splitting (one input set → N output xlsx files
grouped by detected vendor) is deferred to a later milestone — see
[SCOPE.md](SCOPE.md) for the algorithm sketch.

---

## What's NOT built (deferred, each is a clean follow-up commit)

| Deferred | Why deferred | Where it goes when built |
|---|---|---|
| **Advisor probe** (auto-priority from one cheap LLM look at each doc) | Wanted to ship the deterministic merge first; advisor is a cost layer on top. | `worker/advisor.py` + `worker/manifest.py` |
| **PPTX adapter** | LibreOffice dependency check first; user can manually convert in the meantime. | `worker/formats/pptx_adapter.py` |
| **Multi-vendor splitting** | Currently errors with a clear message — strictly safer than guessing. | `worker/run_multi.py` (orchestrator change) |
| **Merged fill-rate analyzer** | Per-doc analysis already runs; cross-doc delta is a reporting nicety. | `analysis/merged_fill_rate.py` |
| **Manifest YAML (per-field wins overrides)** | Priority-only covers the typical case; per-field overrides solve a narrower one. | `worker/manifest.py` |
| **Atomic vocab-cache writes** | One job at a time today, so no race. Becomes important when the cross-job scheduler arrives. | edit to `vocab_map.py` |
| **`run_id` in log lines** | Single-tenant today; only matters when multiple jobs interleave stderr. | edit to logging across the worker |
| **`job_seam.py` stub + README** | Documentation for the future scheduler layer; this file partially covers it. | `worker/job_seam.py` |

---

## Future seams preserved (design rules — don't break these)

These are why the cross-job scheduler will be ~a day of work later, not
a refactor:

1. **`worker.run_multi.run(...)` is a pure function** of `(docs, options)`.
   No globals, no `os.chdir`, no implicit "current PDF."
2. **All per-run state lives under the input folder.** `.pages/`,
   intermediate JSON, output xlsx — all keyed off the input path. So
   multi-tenant = "give each user their own subtree."
3. **Sequential across docs is the default.** Reasons listed above —
   don't flip to parallel-by-default without explicit opt-in.
4. **Worker count comes from `--workers` / env**, never hardcoded. The
   future scheduler will set it per-job from remaining rate budget.
5. **In-process step calls only.** No `subprocess.run("python …")`.
   The future scheduler can run pipelines as async tasks or worker
   processes without a code change.
6. **Don't add cross-doc cache state to module globals.** The
   `vocab_map` on-disk cache is already cross-doc and cross-tenant —
   that's the right boundary.

---

## What was tested before declaring milestone 1 done

1. **All imports resolve clean.** `python -c "from worker import …"`
   succeeds for every submodule.
2. **CLI help renders.** `python -m worker.run_multi --help` works.
3. **Existing single-doc pipeline produces the same output.**
   `python run_pipeline.py files/converse_HO26.pdf --skip classify extract analysis`
   regenerated `BUYSHEET_converse.xlsx` with the same 127 products and
   same 5 flagged pages as before the new code existed. (xlsx zip
   contains an embedded timestamp so byte-diff is misleading; content
   identical.)
4. **Merge logic on real data, no API spend.** Imported `merge_docs`,
   ran against the two existing `.pages/` directories
   (converse + nike), got 127 rows with correct source attribution.
5. **End-to-end merged build, warm cache.**
   `python -m worker.run_multi files/converse_HO26.pdf files/nike_HO26.pdf --vendor X`
   produced a valid xlsx with TEMPLATE + REVIEW sheets, all 127 rows
   written, source-attributed REVIEW entries.
6. **Vendor reconcile guard.** Unit-checked the three cases (single
   vendor → returns it; multi-vendor → SystemExit; override → bypass).
7. **Excel adapter on a synthetic workbook.** Header detection skipped
   blank/title rows, found the header row, dropped a "TOTAL" line via
   the non-SKU pattern, tagged verification correctly.

---

## Don't do

- **Don't refactor [build_buysheet.py](build_buysheet.py).** It works.
  `build_merged.py` reuses its helpers by import. If you need a helper
  factored out, extract it additively (existing CLI behavior unchanged)
  — don't restructure.
- **Don't add per-job globals to `worker/`.** Pure-function discipline
  is what makes the future scheduler trivial. If you need shared state,
  put it on disk (like the vocab cache).
- **Don't make the Excel adapter call the LLM** for the trivial case
  of header → field mapping. The synonym table covers the common 85%
  with zero tokens; only the long tail justifies a Sonnet fallback,
  and even then it should be cached per file-hash.
- **Don't bypass the vendor reconcile.** Silently merging across
  vendors is a correctness bug, not an inconvenience. The `--vendor`
  override exists precisely so the user takes explicit ownership of
  the call.

---

## Quick reference

| Want to … | Do this |
|---|---|
| Add a new format | Drop a `worker/formats/<fmt>_adapter.py` exposing `normalize(path, **kwargs) → pages_dir`. Add the extension to `PDF_EXTS` / `EXCEL_EXTS` in `run_multi.py`. |
| Change merge precedence | Edit `_MERGE_FIELDS` in [worker/merge.py](worker/merge.py). |
| Add a column to the REVIEW sheet | Edit `_write_review_sheet` in [worker/build_merged.py](worker/build_merged.py). |
| Add an Excel header synonym | Add a key to `SYNONYMS` in [worker/formats/excel_adapter.py](worker/formats/excel_adapter.py). |
| Run only normalize (skip merge + build) | Not currently exposed; closest is `python -m worker.formats.pdf_adapter` after adding a `__main__`. Defer until needed. |
