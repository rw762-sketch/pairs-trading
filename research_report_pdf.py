"""Render the project's readable Markdown research report as a paginated PDF.

Usage: venv/bin/python research_report_pdf.py REPORT.md REPORT.pdf
Optional QA: venv/bin/python research_report_pdf.py REPORT.md REPORT.pdf --render-dir tmp/pdfs/review
Compact layout: venv/bin/python research_report_pdf.py REPORT.md REPORT.pdf --profile clean

The report generator owns calculations. This module only formats its Markdown,
tables, links, and local PNG/JPEG figures; it never recalculates financial results.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re
from urllib.parse import unquote

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


BLUE = colors.HexColor("#234c72")
INK = colors.HexColor("#172b3a")
MUTED = colors.HexColor("#556778")
LIGHT = colors.HexColor("#eff4f8")
RULE = colors.HexColor("#d4dfe7")


def _fonts() -> tuple[str, str, str]:
    """Use the project's matplotlib font bundle, including Greek characters."""
    import matplotlib
    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, filename in (
        ("ResearchSans", "DejaVuSans.ttf"),
        ("ResearchSans-Bold", "DejaVuSans-Bold.ttf"),
        ("ResearchSans-Oblique", "DejaVuSans-Oblique.ttf"),
        ("ResearchMono", "DejaVuSansMono.ttf"),
    ):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(folder / filename)))
    pdfmetrics.registerFontFamily(
        "ResearchSans", normal="ResearchSans", bold="ResearchSans-Bold",
        italic="ResearchSans-Oblique", boldItalic="ResearchSans-Bold",
    )
    return "ResearchSans", "ResearchSans-Bold", "ResearchMono"


def _normalize(text: str) -> str:
    return text.translate(str.maketrans({
        "\u2011": "-", "\u2013": "-", "\u2014": " - ", "\u2212": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u00a0": " ", "\u202f": " ",
    }))


def _inline(text: str, base: Path) -> str:
    """Translate a small safe subset of Markdown to ReportLab paragraph tags."""
    text = _normalize(text.strip())
    parts: list[str] = []

    def retain(value: str) -> str:
        parts.append(value)
        return f"ZZINLINE{len(parts) - 1}ZZ"

    def link(match: re.Match) -> str:
        label, destination = match.groups()
        destination = destination.strip().strip("<>")
        if not re.match(r"^[a-zA-Z]+:", destination) and not destination.startswith("#"):
            destination = (base / unquote(destination)).resolve().as_uri()
        return retain(f'<a href="{escape(destination, quote=True)}" color="#234c72">{escape(label)}</a>')

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
    text = re.sub(r"`([^`]+)`", lambda m: retain(f'<font name="ResearchMono">{escape(m[1])}</font>'), text)
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    for index, part in enumerate(parts):
        text = text.replace(f"ZZINLINE{index}ZZ", part)
    return text


def _cells(line: str) -> list[str]:
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def _table(rows: list[list[str]], width: float, styles: dict, base: Path,
           *, profile: str = "standard") -> Table:
    columns = len(rows[0])
    rows = [row + [""] * max(0, columns - len(row)) for row in rows]
    body = styles["TableSmall"] if columns >= 7 else styles["TableBody"]
    header = styles["TableHeadSmall"] if columns >= 7 else styles["TableHead"]
    # Include the header and representative values so textual columns get room.
    scores = []
    for column in range(columns):
        values = [re.sub(r"[*`]", "", row[column]) for row in rows]
        lengths = sorted(len(value) for value in values)
        typical = lengths[min(len(lengths) - 1, int(len(lengths) * 0.85))]
        scores.append(min(31, max(7, typical, len(values[0]) * 0.65)))
    minimum = min(48, width / max(columns, 1) * 0.73)
    spare = width - minimum * columns
    widths = [minimum + spare * score / sum(scores) for score in scores]
    data = [
        [Paragraph(_inline(cell, base), header if index == 0 else body) for cell in row[:columns]]
        for index, row in enumerate(rows)
    ]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    vertical_padding = 4 if profile == "clean" else 6
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), vertical_padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), vertical_padding),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BLUE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
    ]))
    return table


def _styles(profile: str = "standard") -> dict:
    regular, bold, mono = _fonts()
    defaults = getSampleStyleSheet()
    styles = {}
    common = dict(fontName=regular, textColor=INK, splitLongWords=True,
                  allowWidows=False, allowOrphans=False)
    specs = {
        "Body": dict(fontSize=9.4, leading=14.2, spaceAfter=7),
        "Title": dict(fontName=bold, fontSize=24, leading=29, textColor=BLUE, spaceAfter=14, keepWithNext=True),
        "H1": dict(fontName=bold, fontSize=16, leading=20, textColor=BLUE, spaceBefore=17, spaceAfter=8, keepWithNext=True),
        "H2": dict(fontName=bold, fontSize=12, leading=16, textColor=BLUE, spaceBefore=12, spaceAfter=6, keepWithNext=True),
        "H3": dict(fontName=bold, fontSize=10, leading=14, textColor=BLUE, spaceBefore=9, spaceAfter=5, keepWithNext=True),
        "Bullet": dict(fontSize=9.2, leading=13.8, leftIndent=12, firstLineIndent=-8, spaceAfter=4),
        "Caption": dict(fontSize=8.2, leading=11.5, textColor=MUTED, spaceBefore=5, spaceAfter=11),
        "TableBody": dict(fontSize=8.1, leading=11.3, spaceAfter=0),
        "TableHead": dict(fontName=bold, fontSize=8.1, leading=11.3, textColor=colors.white, spaceAfter=0),
        "TableSmall": dict(fontSize=7.3, leading=10.1, spaceAfter=0),
        "TableHeadSmall": dict(fontName=bold, fontSize=7.3, leading=10.1, textColor=colors.white, spaceAfter=0),
        "Code": dict(fontName=mono, fontSize=8, leading=11.5, backColor=LIGHT, borderPadding=8, spaceBefore=4, spaceAfter=9),
    }
    if profile == "clean":
        overrides = {
            "Body": dict(fontSize=9.6, leading=13.6, spaceAfter=6),
            "Title": dict(fontSize=22, leading=26, spaceAfter=11),
            "H1": dict(fontSize=15, leading=18, spaceBefore=8, spaceAfter=6),
            "H2": dict(fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=5),
            "H3": dict(fontSize=10, leading=13, spaceBefore=6, spaceAfter=4),
            "Bullet": dict(fontSize=9.4, leading=13, spaceAfter=3),
            "Caption": dict(fontSize=8.5, leading=11, spaceBefore=4, spaceAfter=7),
            "TableBody": dict(fontSize=9, leading=12),
            "TableHead": dict(fontSize=9, leading=12),
            "TableSmall": dict(fontSize=8.5, leading=11.5),
            "TableHeadSmall": dict(fontSize=8.5, leading=11.5),
        }
        for name, override in overrides.items():
            specs[name].update(override)
    for name, spec in specs.items():
        styles[name] = ParagraphStyle(name, parent=defaults["Normal"], **(common | spec))
    return styles


class _ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        self._bookmark_count = 0

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in {"Title", "H1", "H2"}:
            key = f"section-{self._bookmark_count}"
            self._bookmark_count += 1
            self.canv.bookmarkPage(key)
            # A flat outline remains valid regardless of source heading nesting.
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0, closed=False)


def render_report(markdown_path: str | Path, pdf_path: str | Path,
                  *, profile: str = "standard") -> Path:
    """Render Markdown to A4; ``clean`` fits concise, explicitly paged reports.

    The clean profile uses 9-point table text and compact spacing. A six-column
    table with 26 single-line data rows occupies approximately 540 points, leaving
    room for a heading and short notes. Its figures are capped at 300 points high
    so two spread charts can share a page. Content still wraps or flows onto the
    next page if needed; it is never clipped to enforce a target page count.
    """
    if profile not in {"standard", "clean"}:
        raise ValueError("profile must be 'standard' or 'clean'")
    markdown_path, pdf_path = Path(markdown_path).resolve(), Path(pdf_path).resolve()
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles(profile)
    margin = 43
    doc = _ReportDoc(
        str(pdf_path), pagesize=A4, leftMargin=margin, rightMargin=margin,
        topMargin=51, bottomMargin=46, title="Pairs trading research report",
        author="Pairs trading project", allowSplitting=True,
    )
    width = doc.width

    def page_frame(canvas, _document):
        canvas.saveState()
        canvas.setFont("ResearchSans", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(margin, A4[1] - 29, "PAIRS TRADING  /  RESEARCH REPORT")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(margin, A4[1] - 37, A4[0] - margin, A4[1] - 37)
        canvas.line(margin, 34, A4[0] - margin, 34)
        canvas.drawString(margin, 22, "Historical simulation - exploratory research")
        canvas.drawRightString(A4[0] - margin, 22, f"Page {doc.page}")
        canvas.restoreState()

    frame = Frame(margin, doc.bottomMargin, doc.width, doc.height, leftPadding=0,
                  rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="Research", frames=[frame], onPage=page_frame))
    story = []
    index, title_seen = 0, False
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line in {"<!-- pagebreak -->", "\\newpage"}:
            story.append(PageBreak())
            index += 1
            continue
        if line.startswith("<!--"):
            index += 1
            continue
        if re.match(r"^[-*_]{3,}$", line):
            story.append(Spacer(1, 5))
            index += 1
            continue
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(escape(_normalize(lines[index])))
                index += 1
            story.append(Paragraph("<br/>".join(code), styles["Code"]))
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading:
            level, text = len(heading[1]), heading[2]
            style_name = "Title" if level == 1 and not title_seen else f"H{min(max(level - 1, 1), 3)}"
            title_seen = True
            if style_name == "H1":
                story.append(CondPageBreak(180))
            story.append(Paragraph(_inline(text, markdown_path.parent), styles[style_name]))
            index += 1
            continue
        picture = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if picture:
            caption, relative_path = picture.groups()
            image_path = (markdown_path.parent / unquote(relative_path.strip("<>"))).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Report figure does not exist: {image_path}")
            figure = Image(str(image_path))
            max_image_height = 300 if profile == "clean" else 485
            ratio = min(width / figure.imageWidth, max_image_height / figure.imageHeight)
            figure.drawWidth, figure.drawHeight = figure.imageWidth * ratio, figure.imageHeight * ratio
            figure.hAlign = "CENTER"
            group = [Spacer(1, 5), figure]
            if caption:
                group.append(Paragraph(_inline(caption, markdown_path.parent), styles["Caption"]))
            story.append(KeepTogether(group))
            index += 1
            # Markdown reports may repeat alt text as a visible HTML/Markdown
            # caption. It has already been rendered above; do not print twice.
            following = index
            while following < len(lines) and not lines[following].strip():
                following += 1
            if caption and following < len(lines) and lines[following].strip() == caption:
                index = following + 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[index + 1]):
            rows = [_cells(line)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(_cells(lines[index]))
                index += 1
            story.append(_table(rows, width, styles, markdown_path.parent, profile=profile))
            story.append(Spacer(1, 7 if profile == "clean" else 10))
            continue
        bullet = re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)(.+)", line)
        if bullet:
            text = bullet[1]
            index += 1
            while index < len(lines) and lines[index].startswith("  ") and lines[index].strip():
                text += " " + lines[index].strip()
                index += 1
            story.append(Paragraph("- " + _inline(text, markdown_path.parent), styles["Bullet"]))
            continue
        paragraph = [line.lstrip("> ") if line.startswith(">") else line]
        index += 1
        while index < len(lines) and lines[index].strip():
            following = lines[index].strip()
            if following.startswith(("#", "|", "![", "```", "<!--")) or re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", following):
                break
            paragraph.append(following.lstrip("> ") if following.startswith(">") else following)
            index += 1
        story.append(Paragraph(_inline(" ".join(paragraph), markdown_path.parent), styles["Body"]))
    doc.build(story)
    return pdf_path


def render_pages(pdf_path: str | Path, output_directory: str | Path, dpi: int = 100) -> list[Path]:
    """Rasterize every page for visual QA when system Poppler is unavailable."""
    import pymupdf
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with pymupdf.open(pdf_path) as document:
        for number, page in enumerate(document, start=1):
            path = directory / f"page-{number:02d}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False).save(path)
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown_path", type=Path)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--profile", choices=("standard", "clean"), default="standard")
    args = parser.parse_args()
    print(render_report(args.markdown_path, args.pdf_path, profile=args.profile))
    if args.render_dir:
        print(f"Rendered {len(render_pages(args.pdf_path, args.render_dir))} pages in {args.render_dir}")


if __name__ == "__main__":
    main()
