"""Generate the PDF cahier des charges from its Markdown source.

Usage: python scripts/generate_spec_pdf.py
Requires: reportlab
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "cahier_des_charges.md"
OUTPUT = ROOT / "output" / "pdf" / "cahier-des-charges.pdf"
FONT_DIR = Path("C:/Windows/Fonts")


def register_fonts() -> tuple[str, str]:
    regular = FONT_DIR / "arial.ttf"
    bold = FONT_DIR / "arialbd.ttf"
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("SpecSans", str(regular)))
        pdfmetrics.registerFont(TTFont("SpecSans-Bold", str(bold)))
        pdfmetrics.registerFontFamily("SpecSans", normal="SpecSans", bold="SpecSans-Bold")
        return "SpecSans", "SpecSans-Bold"
    return "Helvetica", "Helvetica-Bold"


def rich(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font color="#116b78">\1</font>', text)
    return text


def draw_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(colors.HexColor("#102a43"))
    canvas.rect(0, height - 14, width, 14, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor("#d6e2e9"))
    canvas.line(48, 45, width - 48, 45)
    canvas.setFillColor(colors.HexColor("#61798c"))
    canvas.setFont("SpecSans" if "SpecSans" in pdfmetrics.getRegisteredFontNames() else "Helvetica", 8)
    canvas.drawString(48, 31, "EMBEDDING LAB  |  Cahier des charges  |  v1.0")
    canvas.drawRightString(width - 48, 31, f"{doc.page}")
    canvas.restoreState()


def main() -> None:
    font, bold_font = register_fonts()
    styles = {
        "title": ParagraphStyle("title", fontName=bold_font, fontSize=21, leading=26, textColor=colors.HexColor("#102a43"), spaceAfter=18),
        "h2": ParagraphStyle("h2", fontName=bold_font, fontSize=12.2, leading=16, textColor=colors.HexColor("#0c6273"), spaceBefore=17, spaceAfter=7, keepWithNext=True),
        "h3": ParagraphStyle("h3", fontName=bold_font, fontSize=10.2, leading=14, textColor=colors.HexColor("#143b52"), spaceBefore=12, spaceAfter=5, keepWithNext=True),
        "body": ParagraphStyle("body", fontName=font, fontSize=9.15, leading=13.4, textColor=colors.HexColor("#213547"), spaceAfter=7),
        "bullet": ParagraphStyle("bullet", fontName=font, fontSize=9.15, leading=13.4, textColor=colors.HexColor("#213547"), leftIndent=14, firstLineIndent=-9, spaceAfter=4),
        "number": ParagraphStyle("number", fontName=font, fontSize=9.15, leading=13.4, textColor=colors.HexColor("#213547"), leftIndent=18, firstLineIndent=-14, spaceAfter=4),
        "meta": ParagraphStyle("meta", fontName=font, fontSize=9.2, leading=14, textColor=colors.HexColor("#476173"), spaceAfter=4),
    }
    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    first_section = next(i for i, line in enumerate(lines) if line.startswith("## "))
    for line in lines[:first_section]:
        if not line.strip():
            continue
        if line.startswith("# "):
            story.append(Paragraph(rich(line[2:]), styles["title"]))
        else:
            story.append(Paragraph(rich(line.strip()), styles["meta"]))
    story.append(Spacer(1, 9))

    buffer = []

    def flush() -> None:
        if buffer:
            story.append(Paragraph(rich(" ".join(buffer)), styles["body"]))
            buffer.clear()

    for line in lines[first_section:]:
        stripped = line.strip()
        if not stripped:
            flush()
        elif stripped.startswith("## "):
            flush()
            story.append(Paragraph(rich(stripped[3:]), styles["h2"]))
        elif stripped.startswith("### "):
            flush()
            story.append(Paragraph(rich(stripped[4:]), styles["h3"]))
        elif stripped.startswith("- "):
            flush()
            story.append(Paragraph("&#8226; " + rich(stripped[2:]), styles["bullet"]))
        elif re.match(r"^\d+\. ", stripped):
            flush()
            number, content = stripped.split(". ", 1)
            story.append(Paragraph(f"{number}. " + rich(content), styles["number"]))
        else:
            buffer.append(stripped)
    flush()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=48, rightMargin=48,
        topMargin=46, bottomMargin=59, title="Cahier des charges - Embedding Lab",
        author="Sacha KEFIF", subject="Semantic Agent Engine",
    )
    frame = Frame(48, 59, A4[0] - 96, A4[1] - 105, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="normal", frames=[frame], onPage=draw_page))
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
