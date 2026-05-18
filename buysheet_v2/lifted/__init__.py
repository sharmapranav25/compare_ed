"""Helpers lifted from the existing kithxkeelo pipeline.

These are battle-tested utilities that survive the architectural cull:
- pdf_render: PyMuPDF/pypdfium2 page rendering at consistent DPI
- photo_embed: openpyxl image embedding + silhouette isolation crop
- vocab_normalize: date/vendor/season normalization against template vocab

Each is self-contained — no dependencies on the old pipeline's relational shape.
"""
