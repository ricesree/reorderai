"""One-off script: parse 'Products unit of sale.pdf' into uom_reference.json.

The PDF is a two-column table (Name, Unit of Measure) repeated across pages.
Each text line extracts as "<product name> <Unit|Weight>". Run this again
whenever the source PDF changes:

    .venv/Scripts/python.exe build_uom_reference.py
"""
import json
import re
from pathlib import Path

from pypdf import PdfReader

SRC = Path(__file__).parent / "Products unit of sale.pdf"
OUT = Path(__file__).parent / "uom_reference.json"

LINE_RE = re.compile(r"^(.*\S)\s+(Unit|Weight)$")


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().upper()


def main():
    reader = PdfReader(str(SRC))
    lookup = {}
    skipped = 0

    for page in reader.pages:
        for line in page.extract_text().splitlines():
            line = line.strip()
            if line in ("Name Unit of Measure", ""):
                continue
            m = LINE_RE.match(line)
            if not m:
                skipped += 1
                continue
            name, uom = m.group(1), m.group(2)
            lookup[normalize(name)] = uom

    OUT.write_text(json.dumps(lookup, indent=2, sort_keys=True))
    print(f"wrote {len(lookup)} products to {OUT.name} ({skipped} unparsed lines skipped)")


if __name__ == "__main__":
    main()
