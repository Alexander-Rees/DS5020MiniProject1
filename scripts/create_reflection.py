"""Build output/pdf/individual_reflection.pdf from docs/individual_reflection.md.

The Markdown file is the editable source. This script reads its title, the bold metadata lines
(Name, Role, Date, Status) and the body paragraphs, then lays them out on one Letter page with
reportlab. It checks that the PDF has one page, that the body is 200 to 260 words and that no
customer-ID-like number appears, and writes a PNG preview to tmp/reflection/ when pdftoppm exists.

    python scripts/create_reflection.py               # build, check, preview
    python scripts/create_reflection.py --no-preview  # skip the PNG preview
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "individual_reflection.md"
OUTPUT = ROOT / "output" / "pdf" / "individual_reflection.pdf"
PREVIEW_DIR = ROOT / "tmp" / "reflection"
BODY_WORDS = (200, 260)
# Customer IDs in this dataset are five-digit numbers from 12346 to 18287.
CUSTOMER_ID = re.compile(r"(?<![\d,.])1[2-8]\d{3}(?:\.0)?(?![\d,])")

TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=15, leading=19, spaceAfter=6)
META = ParagraphStyle("meta", fontName="Helvetica", fontSize=10.5, leading=14, textColor=HexColor("#333333"))
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=11, leading=14.5, spaceAfter=8)


def parse(text):
    """Return (title, [(key, value)], [paragraph]) from the reflection Markdown."""
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    title = blocks[0].lstrip("#").strip()
    meta = re.findall(r"\*\*(.+?):\*\*\s*(.+?)\\?$", blocks[1], flags=re.M)
    if [k for k, _ in meta] != ["Name", "Role", "Date", "Status"]:
        sys.exit(f"expected Name, Role, Date, Status metadata, found {[k for k, _ in meta]}")
    paragraphs = [" ".join(b.split()) for b in blocks[2:]]
    return title, meta, paragraphs


def inline(text):
    """Escape for reportlab and keep Markdown bold."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))


def build(title, meta, paragraphs, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    name = dict(meta)["Name"]
    doc = SimpleDocTemplate(
        str(out), pagesize=LETTER, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
        title=title, author=name, subject="DS5020 Mini-Project 1 individual reflection (draft)",
        creator="scripts/create_reflection.py", invariant=True,
    )
    story = [Paragraph(inline(title), TITLE)]
    story += [Paragraph(f"<b>{escape(k)}:</b> {escape(v)}", META) for k, v in meta]
    story.append(Spacer(1, 12))
    story += [Paragraph(inline(p), BODY) for p in paragraphs]
    doc.build(story)


def check(out, paragraphs):
    pages = len(PdfReader(str(out)).pages)
    words = sum(len(p.split()) for p in paragraphs)
    text = PdfReader(str(out)).pages[0].extract_text()
    problems = []
    if pages != 1:
        problems.append(f"{pages} pages, expected 1")
    if not BODY_WORDS[0] <= words <= BODY_WORDS[1]:
        problems.append(f"body has {words} words, expected {BODY_WORDS[0]} to {BODY_WORDS[1]}")
    if CUSTOMER_ID.search(text):
        problems.append(f"customer-ID-like number found: {CUSTOMER_ID.search(text).group()}")
    return pages, words, problems


def preview(out):
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        print("pdftoppm not found; preview skipped")
        return None
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    stem = PREVIEW_DIR / out.stem
    subprocess.run([pdftoppm, "-png", "-r", "110", "-singlefile", str(out), str(stem)], check=True)
    return stem.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-preview", action="store_true", help="do not render the PNG preview")
    args = parser.parse_args()

    title, meta, paragraphs = parse(SOURCE.read_text(encoding="utf-8"))
    build(title, meta, paragraphs, OUTPUT)
    pages, words, problems = check(OUTPUT, paragraphs)
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {pages} page, {len(paragraphs)} paragraphs, {words} body words")
    if not args.no_preview:
        png = preview(OUTPUT)
        if png:
            print(f"preview {png.relative_to(ROOT)}")
    if problems:
        sys.exit("check failed: " + "; ".join(problems))


if __name__ == "__main__":
    main()
