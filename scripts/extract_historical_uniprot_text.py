"""Compatibility wrapper for historical UniProt extraction.

Historical extraction now lives in scripts/extract_uniprot_text.py.
"""

from __future__ import annotations

import sys

from extract_uniprot_text import main


if __name__ == "__main__":
    raise SystemExit(main(["extract-historical", *sys.argv[1:]]))
