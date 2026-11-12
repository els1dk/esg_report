"""
Generate a FICTIONAL sustainability report for testing and demos.

"Northwind Renewables AG" does not exist and every figure is invented. The
report is built to exercise each part of the pipeline:

  * values in running text, in tables and ONLY in a chart (multimodal test)
  * distractors: targets, Scope 3 categories, intensities, a contents page
  * a page that is an image only (scanned-page test)

Run:  python samples/make_sample_report.py
Needs: reportlab, matplotlib (pip install reportlab matplotlib)
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

OUT = Path(__file__).resolve().parent / "Northwind_Renewables_Sustainability_Report_2024.pdf"
GREEN = colors.HexColor("#1F6F50")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=GREEN, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=GREEN)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5, leading=15)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, textColor=colors.grey)


def table(data, widths=None):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREEN), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F4")]),
    ]))
    return t


def chart_png(title, years, values, ylabel, color):
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=160)
    bars = ax.bar([str(y) for y in years], values, color=color, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01, f"{value:,}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(values) * 1.18)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def scanned_page_png():
    """A text page rendered as a picture, imitating a scanned assurance letter."""
    fig = plt.figure(figsize=(8.27, 11.69), dpi=110)
    fig.text(0.1, 0.9, "Independent Limited Assurance Report", fontsize=17, weight="bold")
    lines = [
        "To the Supervisory Board of Northwind Renewables AG (fictional).",
        "",
        "We performed a limited assurance engagement (ISAE 3000 Revised) on",
        "selected 2024 sustainability information: Scope 1 and Scope 2 GHG",
        "emissions, energy consumption and the share of women in management.",
        "",
        "Based on the procedures performed, nothing has come to our attention",
        "that causes us to believe that the selected information is not",
        "prepared, in all material respects, in accordance with the criteria.",
        "",
        "Example Assurance GmbH (fictional), Hamburg, 14 March 2025",
    ]
    for i, line in enumerate(lines):
        fig.text(0.1, 0.84 - i * 0.03, line, fontsize=12, family="serif")
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def build():
    story = []
    # 1 – cover
    story += [Spacer(1, 5 * cm), Paragraph("Northwind Renewables AG", H1),
              Paragraph("Sustainability Report 2024", ParagraphStyle("t", parent=H1, fontSize=26, leading=32)),
              Spacer(1, 1 * cm),
              Paragraph("Reporting period: 1 January – 31 December 2024. Prepared with reference to the "
                        "GRI Standards and the ESRS (CSRD).", BODY),
              Spacer(1, 6 * cm),
              Paragraph("FICTIONAL SAMPLE DOCUMENT – the company and all figures are invented. It was "
                        "generated to test the ESG Report Intelligence application.", SMALL),
              PageBreak()]
    # 2 – contents (navigation page that mentions every topic)
    story += [Paragraph("Contents", H1)]
    for title, page in [("Letter from the CEO", 3), ("Climate: Scope 1, Scope 2 and Scope 3 emissions", 4),
                        ("Energy and water consumption", 5), ("Waste and recycling", 6),
                        ("Our people: women in management, board diversity, parental leave", 7),
                        ("Supply chain: child labour and forced labour audits", 8),
                        ("Independent assurance", 9), ("GRI content index", 10)]:
        story.append(Paragraph(f"{title} {'.' * 40} {page}", BODY))
    story.append(PageBreak())
    # 3 – CEO letter with targets (distractors)
    story += [Paragraph("Letter from the CEO", H1),
              Paragraph("Dear stakeholders, 2024 was a year of steady progress. We cut our direct emissions "
                        "for the third year in a row and expanded on-site solar generation at four plants.", BODY),
              Paragraph("Our climate target remains ambitious: we aim to reduce Scope 1 and Scope 2 emissions "
                        "by 50% by 2030 compared with 2022, and we commit to net zero by 2045. We also aim to "
                        "have 40% women in management positions by 2028.", BODY),
              Paragraph("Thank you for your trust. — Dr. Lena Brandt, Chief Executive Officer (fictional)", BODY),
              PageBreak()]
    # 4 – climate (text + table + distractor categories / intensity)
    story += [Paragraph("Climate: greenhouse gas emissions", H1),
              Paragraph("In 2024, direct (Scope 1) GHG emissions amounted to 41,250 tCO2e (2023: 44,870 tCO2e). "
                        "Scope 2 emissions were 12,430 tCO2e using the market-based method and 18,960 tCO2e "
                        "using the location-based method. Total Scope 3 emissions were 389,600 tCO2e, of which "
                        "purchased goods and services (category 1) accounted for 251,300 tCO2e.", BODY),
              Spacer(1, 0.3 * cm),
              Paragraph("Table 1 – GHG emissions (t CO2e)", H2),
              table([["GHG emissions (t CO2e)", "2022", "2023", "2024"],
                     ["Scope 1 (direct)", "47,310", "44,870", "41,250"],
                     ["Scope 2 (market-based)", "15,980", "14,120", "12,430"],
                     ["Scope 2 (location-based)", "19,870", "19,410", "18,960"],
                     ["Scope 3 total", "412,300", "401,750", "389,600"],
                     ["  thereof cat. 1 purchased goods", "262,400", "258,900", "251,300"],
                     ["GHG intensity (t CO2e per EUR m revenue)", "38.2", "34.9", "31.6"]],
                    widths=[8.5 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]),
              Spacer(1, 0.4 * cm),
              Paragraph("Emissions were calculated in line with the GHG Protocol Corporate Standard using an "
                        "operational control approach.", SMALL),
              PageBreak()]
    # 5 – energy and water (water values ONLY in the chart)
    story += [Paragraph("Energy and water", H1),
              Paragraph("Total energy consumption within the organisation was 286,400 MWh in 2024. The share "
                        "of renewable electricity reached 78%, up from 71% in 2023.", BODY),
              Paragraph("Water consumption continued to fall thanks to closed-loop cooling (see Figure 3). "
                        "In 2024, 94% of discharged wastewater was treated before being returned to the "
                        "environment.", BODY),
              Image(chart_png("Figure 3 – Water consumption (thousand m³)", [2022, 2023, 2024],
                              [1320, 1245, 1180], "thousand m³", "#3A8DBF"), width=15 * cm, height=7.5 * cm),
              PageBreak()]
    # 6 – waste
    story += [Paragraph("Waste and recycling", H1),
              Paragraph("Total waste generated was 22,020 t in 2024. Of this, 18,940 t were recycled or reused "
                        "and thereby diverted from disposal, which corresponds to 86% of total waste. Hazardous "
                        "waste amounted to 1,130 t.", BODY),
              Image(chart_png("Figure 4 – Waste diverted from disposal (t)", [2022, 2023, 2024],
                              [16880, 17920, 18940], "tonnes", "#5BA36F"), width=15 * cm, height=7.5 * cm),
              PageBreak()]
    # 7 – people
    story += [Paragraph("Our people", H1),
              Paragraph("At year end 2024, Northwind employed 3,412 people (headcount).", BODY),
              Paragraph("Table 2 – Diversity indicators", H2),
              table([["Indicator", "2023", "2024"],
                     ["Women in management (%)", "32.8", "34.2"],
                     ["Women in leadership – senior executives (%)", "25.0", "27.5"],
                     ["Women on the Supervisory Board (%)", "41.7", "41.7"],
                     ["Lost-time injury frequency rate (per million hours)", "2.1", "1.8"]],
                    widths=[10 * cm, 2.6 * cm, 2.6 * cm]),
              Spacer(1, 0.3 * cm),
              Paragraph("Five of the twelve Supervisory Board members are women. The women-to-men average "
                        "salary ratio was 0.94 for the total workforce and 0.91 for management.", BODY),
              Paragraph("Maternity leave is fully paid for 16 weeks and is available to 100% of employees in "
                        "all countries. Paternity leave is fully paid for 4 weeks for 100% of employees.", BODY),
              PageBreak()]
    # 8 – supply chain
    story += [Paragraph("Supply chain responsibility", H1),
              Paragraph("In 2024 we carried out 412 supplier audits covering 61% of procurement spend. "
                        "No suppliers with child labour violations were identified (0 cases). Two suppliers with "
                        "forced labour violations were identified; both were suspended pending remediation.", BODY),
              PageBreak()]
    # 9 – scanned assurance page (image only, no text layer)
    story += [Image(scanned_page_png(), width=17 * cm, height=24 * cm), PageBreak()]
    # 10 – GRI index (navigation page)
    story += [Paragraph("GRI content index", H1),
              table([["GRI", "Disclosure", "Page"],
                     ["GRI 305-1", "Direct (Scope 1) GHG emissions", "4"],
                     ["GRI 305-2", "Energy indirect (Scope 2) GHG emissions", "4"],
                     ["GRI 305-3", "Other indirect (Scope 3) GHG emissions", "4"],
                     ["GRI 302-1", "Energy consumption within the organization", "5"],
                     ["GRI 303-5", "Water consumption", "5"],
                     ["GRI 306-4", "Waste diverted from disposal", "6"],
                     ["GRI 405-1", "Diversity of governance bodies and employees", "7"],
                     ["GRI 401-3", "Parental leave", "7"],
                     ["GRI 408-1", "Operations and suppliers at significant risk for child labor", "8"],
                     ["GRI 409-1", "Operations and suppliers at significant risk for forced labor", "8"]],
                    widths=[2.5 * cm, 11 * cm, 2 * cm])]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Northwind Renewables AG – Sustainability Report 2024 (fictional sample)",
                            author="Northwind Renewables AG (fictional)")
    doc.build(story)
    print(f"Written: {OUT}")


if __name__ == "__main__":
    build()
