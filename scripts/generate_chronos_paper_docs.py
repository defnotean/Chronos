"""Generate publication-grade Word (.docx) and PDF (.pdf) editions of the Chronos Architecture Paper.

Outputs:
  - docs/CHRONOS_ARCHITECTURE_PAPER.docx
"""
from __future__ import annotations

import re
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "CHRONOS_ARCHITECTURE_PAPER.md"
DOCX_PATH = ROOT / "docs" / "CHRONOS_ARCHITECTURE_PAPER.docx"
PDF_PATH = ROOT / "docs" / "CHRONOS_ARCHITECTURE_PAPER.pdf"
FIG_PATH = ROOT / "assets" / "chronos_1t_gb10_a100_proof.png"

HEX_PRIMARY = "0F172A"
HEX_SECONDARY = "1E293B"
HEX_MUTED = "475569"
HEX_ROW_ZEBRA = "F8FAFC"
HEX_BORDER = "CBD5E1"
HEX_CODE_BG = "F1F5F9"

COLOR_PRIMARY = RGBColor(15, 23, 42)
COLOR_SECONDARY = RGBColor(30, 41, 59)
COLOR_BODY = RGBColor(51, 65, 85)
COLOR_MUTED = RGBColor(100, 116, 139)
COLOR_ACCENT = RGBColor(6, 95, 70)


def set_cell_bg(cell, hex_color: str) -> None:
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_margins(cell, top=80, bottom=80, left=120, right=120) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def set_table_borders(table, color=HEX_BORDER) -> None:
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def clean_md_text(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    return text.strip()


def add_formatted_paragraph(doc, line: str, bullet: bool = False):
    p = doc.add_paragraph(style="List Bullet" if bullet else None)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    # Replace markdown links [label](url) -> label (url)
    line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", line)
    parts = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", line)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2])
            run.bold = True
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)
            run.font.color.rgb = COLOR_PRIMARY
        elif part.startswith("`") and part.endswith("`"):
            run = p.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
            run.font.color.rgb = COLOR_ACCENT
        else:
            run = p.add_run(part)
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)
            run.font.color.rgb = COLOR_BODY
    return p


def build_docx() -> None:
    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False
                tbl = doc.add_table(rows=1, cols=1)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                set_table_borders(tbl)
                cell = tbl.cell(0, 0)
                set_cell_bg(cell, HEX_CODE_BG)
                set_cell_margins(cell, 100, 100, 140, 140)
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                run = p.add_run("\n".join(code_lines))
                run.font.name = "Consolas"
                run.font.size = Pt(8.0)
                run.font.color.rgb = COLOR_PRIMARY
                doc.add_paragraph()
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not line.strip() or line.strip() == "---":
            i += 1
            continue

        if line.strip().startswith("!["):
            if FIG_PATH.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                run.add_picture(str(FIG_PATH), width=Inches(6.8))
            i += 1
            continue

        if line.startswith("# "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            run = p.add_run(clean_md_text(line[2:]))
            run.bold = True
            run.font.name = "Calibri"
            run.font.size = Pt(18)
            run.font.color.rgb = COLOR_PRIMARY
            i += 1
            continue

        if line.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(clean_md_text(line[3:]))
            run.bold = True
            run.font.name = "Calibri"
            run.font.size = Pt(14)
            run.font.color.rgb = COLOR_PRIMARY
            i += 1
            continue

        if line.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(clean_md_text(line[4:]))
            run.bold = True
            run.font.name = "Calibri"
            run.font.size = Pt(11.5)
            run.font.color.rgb = COLOR_SECONDARY
            i += 1
            continue

        # Markdown Table
        if line.strip().startswith("|"):
            table_rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row_str = lines[i].strip().strip("|")
                cells = [c.strip() for c in row_str.split("|")]
                if not all(set(c) <= {":", "-", " "} for c in cells):
                    table_rows.append(cells)
                i += 1
            if table_rows:
                ncols = max(len(r) for r in table_rows)
                tbl = doc.add_table(rows=len(table_rows), cols=ncols)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                set_table_borders(tbl)
                for r_idx, r_data in enumerate(table_rows):
                    for c_idx in range(ncols):
                        val = clean_md_text(r_data[c_idx]) if c_idx < len(r_data) else ""
                        cell = tbl.cell(r_idx, c_idx)
                        set_cell_margins(cell, 60, 60, 90, 90)
                        if r_idx == 0:
                            set_cell_bg(cell, HEX_PRIMARY)
                        elif r_idx % 2 == 1:
                            set_cell_bg(cell, HEX_ROW_ZEBRA)
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_after = Pt(0)
                        run = p.add_run(val)
                        run.font.name = "Calibri"
                        run.font.size = Pt(8.5 if ncols >= 6 else 9.0)
                        if r_idx == 0:
                            run.bold = True
                            run.font.color.rgb = RGBColor(255, 255, 255)
                        else:
                            if c_idx == 0:
                                run.bold = True
                            run.font.color.rgb = COLOR_BODY
                doc.add_paragraph()
            continue

        if line.lstrip().startswith(("- ", "* ")):
            add_formatted_paragraph(doc, line.lstrip()[2:], bullet=True)
            i += 1
            continue

        if re.match(r"^\d+\.\s+", line.lstrip()):
            add_formatted_paragraph(doc, line.lstrip(), bullet=False)
            i += 1
            continue

        add_formatted_paragraph(doc, line, bullet=False)
        i += 1

    DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(DOCX_PATH))
    print(f"Saved {DOCX_PATH} ({DOCX_PATH.stat().st_size:,} bytes)")

    try:
        import win32com.client

        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        try:
            wdoc = word.Documents.Open(str(DOCX_PATH.resolve()))
            try:
                wdoc.SaveAs(str(PDF_PATH.resolve()), FileFormat=17)
            finally:
                wdoc.Close(False)
        finally:
            word.Quit()
        print(f"Saved {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")
    except Exception as exc:
        print(f"PDF export via Word COM warning: {exc}")


if __name__ == "__main__":
    build_docx()
