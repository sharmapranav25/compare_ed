/**
 * Kith buy-sheet image fixer.
 *
 * Convert floating image overlays (OverGridImage — what openpyxl writes as
 * xlsx OneCellAnchor) into in-cell CellImages so Google Sheets binds the
 * photo to its row and never drifts.
 *
 * How: each floating image's bytes already live in the spreadsheet (Google
 * extracted them on xlsx import). We read those bytes via getBlob(),
 * base64-encode them, hand them to newCellImage().setSourceUrl() as a
 * `data:image/png;base64,...` URI, and write the result into the anchor
 * cell. Then delete the original float. The bytes never leave the sheet —
 * no external URL, no public hosting.
 *
 * Per-cell hard limit: GS cell values cap at 50,000 characters. Base64
 * inflates by ~4/3, so anything over ~37KB raw can't fit. The script
 * leaves oversized images alone and reports the count in the toast.
 *
 * Idempotent. After conversion CellImages don't appear in getImages(), so
 * re-running on the same sheet does nothing.
 */

const MAX_BYTES_FOR_INLINE = 37000;  // ~50k chars after base64

function onOpen() {
  SpreadsheetApp.getUi().createMenu('Buy Sheet')
    .addItem('Fix images after import', 'fixImagesInPlace')
    .addToUi();
}

function fixImagesInPlace() {
  const ss = SpreadsheetApp.getActive();
  let converted = 0;
  let skippedTooLarge = 0;
  let skippedNoAnchor = 0;
  let errors = 0;

  for (const sheet of ss.getSheets()) {
    // Snapshot the image list — we mutate the sheet while iterating.
    const images = sheet.getImages();
    for (const img of images) {
      try {
        const blob = img.getBlob();
        const bytes = blob.getBytes();
        if (bytes.length > MAX_BYTES_FOR_INLINE) {
          skippedTooLarge++;
          continue;
        }
        const anchor = img.getAnchorCell();
        if (!anchor) {
          skippedNoAnchor++;
          continue;
        }
        const mime = blob.getContentType() || 'image/png';
        const dataUri = 'data:' + mime + ';base64,' + Utilities.base64Encode(bytes);
        const cellImage = SpreadsheetApp.newCellImage()
          .setSourceUrl(dataUri)
          .setAltTextTitle('Product photo')
          .build();
        anchor.setValue(cellImage);
        img.remove();
        converted++;
      } catch (e) {
        Logger.log('fixImagesInPlace error: ' + e.message);
        errors++;
      }
    }
  }

  const parts = ['Converted ' + converted + ' images to in-cell'];
  if (skippedTooLarge) parts.push('skipped ' + skippedTooLarge + ' >37KB');
  if (skippedNoAnchor) parts.push('skipped ' + skippedNoAnchor + ' no anchor');
  if (errors) parts.push(errors + ' errors (see Logger)');
  ss.toast(parts.join('; '), 'Buy Sheet', 10);
}
