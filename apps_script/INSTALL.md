# Apps Script: Fix Floating Images in Google Sheets

## Why this exists

The buy-sheet pipeline produces xlsx files with product photos embedded as
floating images. Google Sheets imports them, then auto-collapses any cell
whose text content is empty — which photo cells are — so the floats drift
out of their intended rows.

`fix_images.gs` converts each floating image into an **in-cell CellImage**
by reading its bytes (already inside the spreadsheet — Google extracted
them on import), base64-encoding them inline, and writing them into the
anchor cell via `SpreadsheetApp.newCellImage().setSourceUrl("data:image/png;base64,...")`.
The bytes never leave the spreadsheet. No external hosting, no public URLs.

## One-time install

1. Create a Google Sheet to use as your buy-sheet **workbench** (any name).
2. `Extensions → Apps Script` (opens the editor in a new tab).
3. Delete the default `myFunction` stub.
4. Paste the contents of [fix_images.gs](fix_images.gs).
5. **Save** (floppy-disk icon or Cmd-S). The project name doesn't matter.
6. Close the Apps Script tab.

The menu `Buy Sheet → Fix images after import` will appear automatically
every time you open this spreadsheet.

## Per-catalog use

For each new buy sheet you import:

1. Open the workbench. (Or: `File → Make a copy` first if you want to keep
   prior catalogs around — Apps Script copies over with the spreadsheet.)
2. `File → Import → Upload → BUYSHEET_<vendor>.xlsx → Replace spreadsheet`.
   Google Sheets reloads the tab; the menu re-appears on its own.
3. Click `Buy Sheet → Fix images after import`.
4. **First-run only:** authorize the script.
   "This app isn't verified" is normal — Apps Script tied to a single
   spreadsheet always shows this. Click `Advanced → Go to Untitled project
   (unsafe) → Allow`. The script accesses only this one spreadsheet; no
   Drive, no network.
5. Watch the toast: `Converted N images to in-cell`. Scroll the sheet —
   every photo is now bound to its row.

## Troubleshooting

- **`skipped N >37KB`** — those PNGs exceed the per-cell 50,000-char value
  limit (base64 inflates bytes by ~33%). They stay floating. Excel renders
  them correctly; use Excel for those rows, or shrink the source PNGs.
- **`skipped N no anchor`** — image not bound to any cell. Usually means
  the xlsx writer changed format; report it.
- **`0 images`** — either no floats present (already converted, or import
  didn't include images) or the script already ran (idempotent).
- **Menu doesn't appear** — reload the spreadsheet tab. `onOpen` fires on
  document load.
- **First-run permission dialog scary** — yes, Google flags unverified
  bound scripts. Required scopes are only Spreadsheet read/write on THIS
  document. No external services.

## What this script does NOT do

- Does not contact any external service.
- Does not modify cell values other than photo cells (col A in the Kith
  template).
- Does not change row heights, column widths, or other formatting.
- Does not affect the original xlsx (still has floating images, renders
  correctly in Excel without any conversion step).
