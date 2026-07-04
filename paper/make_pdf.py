"""Render the paper markdown to PDF (fpdf2 + markdown, no system deps).

Usage: python paper/make_pdf.py paper/BHDCMoralLM_paper.md paper/BHDCMoralLM_paper.pdf
"""
from __future__ import annotations

import sys
import re
import markdown as md
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def render(md_path: str, pdf_path: str) -> None:
    text = open(md_path, encoding="utf-8").read()

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    # full-Unicode fonts (em-dashes, ×, ≈, ≥, •, −, λ, ν ... all appear in the paper)
    D = "/usr/share/fonts/truetype/dejavu"
    pdf.add_font("DV", "", f"{D}/DejaVuSans.ttf")
    pdf.add_font("DV", "B", f"{D}/DejaVuSans-Bold.ttf")
    pdf.add_font("DVM", "", f"{D}/DejaVuSansMono.ttf")
    pdf.add_page()
    pdf.set_margins(18, 16, 18)

    def line(txt, size=11, style="", gap=1.4, color=(20, 20, 20)):
        pdf.set_font("DV", style, size)
        pdf.set_text_color(*color)
        pdf.multi_cell(0, gap * size / 2.4, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def rule():
        pdf.ln(1.5)
        y = pdf.get_y()
        pdf.set_draw_color(200, 200, 200)
        pdf.line(18, y, pdf.w - 18, y)
        pdf.ln(2.5)

    in_code = False
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal table_rows
        if not table_rows:
            return
        rows = [r for r in table_rows if not re.match(r"^\s*\|?[\s:|-]+\|?\s*$", r)]
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        if not cells:
            table_rows = []
            return
        ncol = max(len(r) for r in cells)
        avail = pdf.w - 36
        cw = avail / ncol
        pdf.set_font("DV", "", 8.5)
        for ri, row in enumerate(cells):
            style = "B" if ri == 0 else ""
            pdf.set_font("DV", style, 8.5)
            row = row + [""] * (ncol - len(row))
            # measure row height
            h = 4.6
            x0 = pdf.get_x(); y0 = pdf.get_y()
            if y0 + h > pdf.h - 16:
                pdf.add_page(); x0 = pdf.get_x(); y0 = pdf.get_y()
            for ci, c in enumerate(row):
                pdf.set_xy(18 + ci * cw, y0)
                pdf.set_fill_color(238, 238, 238) if ri == 0 else pdf.set_fill_color(255, 255, 255)
                pdf.multi_cell(cw, h, c, border=1, align="L", fill=(ri == 0),
                               new_x=XPos.RIGHT, new_y=YPos.TOP, max_line_height=4.2)
            pdf.set_xy(18, y0 + h)
        pdf.ln(2)
        table_rows = []

    for raw in text.split("\n"):
        s = raw.rstrip()
        if s.strip().startswith("```"):
            in_code = not in_code
            if not in_code:
                pdf.ln(1)
            continue
        if in_code:
            pdf.set_font("DVM", "", 8.0)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 4, s if s else " ", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            continue
        if s.strip().startswith("|"):
            in_table = True
            table_rows.append(s)
            continue
        elif in_table:
            flush_table()
            in_table = False
        if not s.strip():
            pdf.ln(1.6)
            continue
        if s.startswith("# "):
            pdf.ln(1); line(s[2:], 17, "B", color=(10, 10, 40)); rule()
        elif s.startswith("## "):
            pdf.ln(1.5); line(s[3:], 13, "B", color=(20, 20, 60))
        elif s.startswith("### "):
            line(s[4:], 11.5, "B", color=(30, 30, 70))
        elif s.strip() in ("---", "***"):
            rule()
        else:
            # strip inline markdown emphasis/backticks/links for plain text
            body = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
            body = re.sub(r"\*(.+?)\*", r"\1", body)
            body = re.sub(r"`(.+?)`", r"\1", body)
            body = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", body)
            indent = ""
            if body.lstrip().startswith(("- ", "* ")):
                body = "  • " + body.lstrip()[2:]
            line(body, 10.5)
    if in_table:
        flush_table()

    pdf.output(pdf_path)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
