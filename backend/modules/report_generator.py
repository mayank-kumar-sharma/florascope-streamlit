"""
PDF Report Generator
Generates a Pre-Feasibility GIS Eligibility Screening Report
matching the Flora Carbon AI report format.
"""

import io
import os
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────
# COLOR SCHEME
# ──────────────────────────────────────────────
BRAND_GREEN = colors.HexColor("#1a9850")
BRAND_DARK = colors.HexColor("#1a1a2e")
BRAND_LIGHT = colors.HexColor("#f0f4f0")
STATUS_COLORS = {
    "Highly Suitable": colors.HexColor("#1a9850"),
    "Suitable": colors.HexColor("#66bd63"),
    "Possibly Suitable": colors.HexColor("#fee08b"),
    "Not Suitable": colors.HexColor("#d73027"),
    "Needs Further Investigation": colors.HexColor("#fdae61"),
}

LULC_COLORS = {
    "Trees": "#1a9850",
    "Mangroves": "#006400",
    "Crops": "#e8d354",
    "Rangeland": "#a0c93d",
    "Grass": "#88B053",
    "Shrub & Scrub": "#DFC35A",
    "Bare Ground": "#a39171",
    "Built Area": "#d13c1b",
    "Water": "#0096ff",
    "Flooded Vegetation": "#7a87c6",
    "Snow/Ice": "#b5e7ff",
    "No Data": "#cccccc",
}


def create_styles():
    """Create custom paragraph styles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=BRAND_DARK,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.gray,
        alignment=TA_CENTER,
        spaceAfter=20,
    ))

    styles.add(ParagraphStyle(
        name="SectionHeader",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=BRAND_GREEN,
        spaceBefore=16,
        spaceAfter=8,
        fontName="Helvetica-Bold",
        borderWidth=1,
        borderColor=BRAND_GREEN,
        borderPadding=4,
    ))

    styles.add(ParagraphStyle(
        name="SubSection",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=BRAND_DARK,
        spaceBefore=10,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="BodyText2",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="SmallText",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.gray,
    ))

    styles.add(ParagraphStyle(
        name="StatusGood",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#1a9850"),
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="StatusBad",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#d73027"),
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        name="Warning",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#e65100"),
        leftIndent=12,
    ))

    return styles


def generate_lulc_pie_chart(lulc_data: Dict, title: str, output_path: str):
    """Generate a LULC pie chart as PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    labels = []
    sizes = []
    chart_colors = []

    for cls, area in sorted(lulc_data.items(), key=lambda x: x[1], reverse=True):
        if area > 0:
            labels.append(f"{cls}\n({area:.1f} ha)")
            sizes.append(area)
            chart_colors.append(LULC_COLORS.get(cls, "#cccccc"))

    if sizes:
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=chart_colors,
            autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 7}
        )
        for autotext in autotexts:
            autotext.set_fontsize(6)
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_forest_loss_chart(loss_by_year: Dict, output_path: str):
    """Generate forest loss by year bar chart."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))

    years = sorted(loss_by_year.keys())
    values = [loss_by_year[y] for y in years]

    ax.bar(years, values, color="#d73027", alpha=0.8, width=0.8)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Forest Loss (ha)", fontsize=8)
    ax.set_title("Annual Forest Cover Loss (Hansen)", fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_ndvi_chart(ndvi_data: Dict, output_path: str):
    """Generate NDVI time series chart."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))

    years = sorted(ndvi_data.keys())
    values = [ndvi_data[y] if ndvi_data[y] is not None else 0 for y in years]

    ax.plot(years, values, "g-o", linewidth=2, markersize=5)
    ax.fill_between(years, values, alpha=0.2, color="green")
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Mean NDVI", fontsize=8)
    ax.set_title("Vegetation Health (NDVI) Over Time", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_eligibility_chart(results: List[Dict], output_path: str):
    """Generate eligibility comparison chart."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    types = [r["project_type"] for r in results]
    scores = [r["confidence"] for r in results]
    bar_colors = []
    for r in results:
        status = r["status"]
        if status == "Highly Suitable":
            bar_colors.append("#1a9850")
        elif status == "Suitable":
            bar_colors.append("#66bd63")
        elif status == "Possibly Suitable":
            bar_colors.append("#fee08b")
        else:
            bar_colors.append("#d73027")

    bars = ax.barh(types, scores, color=bar_colors, height=0.5, edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Eligibility Score (%)", fontsize=9)
    ax.set_title("Project Type Eligibility Comparison", fontsize=11, fontweight="bold")

    for bar, score in zip(bars, scores):
        ax.text(score + 2, bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}%", va="center", fontsize=9, fontweight="bold")

    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(analysis_results: Dict, eligibility_results: Dict,
                    output_path: str, project_name: str = "Project Site") -> str:
    """
    Generate full pre-feasibility PDF report.
    """
    styles = create_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
    )

    elements = []
    meta = analysis_results.get("metadata", {})
    land = eligibility_results.get("land_profile", {})
    results_list = eligibility_results.get("results", [])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COVER / HEADER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(
        width="100%", thickness=3, color=BRAND_GREEN, spaceAfter=10
    ))
    elements.append(Paragraph(
        "Pre-Feasibility GIS Eligibility<br/>Screening Report",
        styles["ReportTitle"]
    ))
    elements.append(Paragraph(
        f"{project_name} | Generated {datetime.now().strftime('%B %d, %Y')}",
        styles["ReportSubtitle"]
    ))
    elements.append(HRFlowable(
        width="100%", thickness=3, color=BRAND_GREEN, spaceBefore=5, spaceAfter=20
    ))

    # Summary box
    rec = eligibility_results.get("recommended_project_type", "N/A")
    conf = eligibility_results.get("overall_confidence", 0)
    summary_data = [
        ["Analysis Period", meta.get("analysis_period", "N/A")],
        ["Location", f"{meta.get('centroid_lat', 0):.4f}°N, {meta.get('centroid_lon', 0):.4f}°E"],
        ["Total Area", f"{land.get('total_area_ha', 0):.1f} ha"],
        ["Recommended Project Type", f"<b>{rec}</b>"],
        ["Overall Confidence Score", f"<b>{conf:.0f}%</b>"],
    ]

    summary_table = Table(
        [[Paragraph(r[0], styles["BodyText2"]),
          Paragraph(str(r[1]), styles["BodyText2"])] for r in summary_data],
        colWidths=[150, 300],
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), BRAND_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 15))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 1: CURRENT LAND USE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(Paragraph("1. Current Land Use Classification", styles["SectionHeader"]))

    current_lu = land.get("current_land_use", {})
    lu_rows = [
        [Paragraph("<b>Land Use Class</b>", styles["BodyText2"]),
         Paragraph("<b>Coverage (%)</b>", styles["BodyText2"]),
         Paragraph("<b>Area (ha)</b>", styles["BodyText2"])],
    ]
    total_ha = land.get("total_area_ha", 1)
    for cls, pct in sorted(current_lu.items(), key=lambda x: x[1], reverse=True):
        if pct > 0:
            area = (pct / 100) * total_ha
            lu_rows.append([
                Paragraph(cls.replace("_pct", "").replace("_", " ").title(), styles["BodyText2"]),
                Paragraph(f"{pct:.1f}%", styles["BodyText2"]),
                Paragraph(f"{area:.1f}", styles["BodyText2"]),
            ])

    lu_table = Table(lu_rows, colWidths=[200, 100, 100])
    lu_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(lu_table)
    elements.append(Spacer(1, 10))

    # LULC pie chart (latest year)
    lulc_ts = analysis_results.get("lulc_timeseries", {})
    years_sorted = sorted(lulc_ts.keys(), reverse=True)
    chart_dir = os.path.dirname(output_path)

    if years_sorted:
        latest_lulc = lulc_ts[years_sorted[0]]
        if latest_lulc:
            pie_path = os.path.join(chart_dir, "lulc_pie_latest.png")
            generate_lulc_pie_chart(latest_lulc, f"Land Use Distribution ({years_sorted[0]})", pie_path)
            elements.append(Image(pie_path, width=350, height=280))
            elements.append(Spacer(1, 10))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 2: HISTORICAL ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(PageBreak())
    elements.append(Paragraph("2. Historical Land Use & Forest Change Analysis", styles["SectionHeader"]))

    hist = land.get("historical", {})
    hist_rows = [
        ["Forest Cover in Year 2000 (Hansen)", f"{hist.get('hansen_forest_2000_pct', 0):.1f}%"],
        ["Forest 10 Years Ago", f"{hist.get('forest_pct_10yr_ago', 0):.1f}%"],
        ["Was Forest 10yr Ago", "Yes" if hist.get("was_forest_10yr_ago") else "No"],
        ["Deforestation Detected", "Yes" if hist.get("has_deforestation") else "No"],
        ["Total Forest Loss (Hansen)", f"{hist.get('deforestation_ha', 0):.1f} ha"],
        ["Consecutive Years Non-Forest", str(hist.get("years_non_forest", 0))],
    ]

    hist_table = Table(
        [[Paragraph(r[0], styles["BodyText2"]),
          Paragraph(str(r[1]), styles["BodyText2"])] for r in hist_rows],
        colWidths=[250, 200],
    )
    hist_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), BRAND_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(hist_table)
    elements.append(Spacer(1, 10))

    # Forest loss chart
    hansen = analysis_results.get("hansen_forest", {})
    loss_by_year = hansen.get("loss_by_year", {})
    if loss_by_year:
        loss_chart_path = os.path.join(chart_dir, "forest_loss_chart.png")
        generate_forest_loss_chart(loss_by_year, loss_chart_path)
        elements.append(Image(loss_chart_path, width=450, height=200))
        elements.append(Spacer(1, 10))

    # NDVI chart
    ndvi_data = analysis_results.get("ndvi_timeseries", {})
    if ndvi_data:
        ndvi_path = os.path.join(chart_dir, "ndvi_chart.png")
        generate_ndvi_chart(ndvi_data, ndvi_path)
        elements.append(Image(ndvi_path, width=450, height=200))
        elements.append(Spacer(1, 10))

    # ── 10-year LULC comparison (earliest vs latest) ──
    if len(years_sorted) >= 2:
        elements.append(Paragraph("2.1 Land Use Change Comparison", styles["SubSection"]))

        earliest_lulc = lulc_ts[years_sorted[-1]]
        latest_lulc = lulc_ts[years_sorted[0]]
        all_classes = set(list(earliest_lulc.keys()) + list(latest_lulc.keys()))

        change_rows = [
            [Paragraph("<b>Land Class</b>", styles["BodyText2"]),
             Paragraph(f"<b>{years_sorted[-1]} (ha)</b>", styles["BodyText2"]),
             Paragraph(f"<b>{years_sorted[0]} (ha)</b>", styles["BodyText2"]),
             Paragraph("<b>Change (ha)</b>", styles["BodyText2"])],
        ]
        for cls in sorted(all_classes):
            early = earliest_lulc.get(cls, 0)
            late = latest_lulc.get(cls, 0)
            change = late - early
            change_str = f"+{change:.1f}" if change > 0 else f"{change:.1f}"
            change_rows.append([
                Paragraph(cls, styles["BodyText2"]),
                Paragraph(f"{early:.1f}", styles["BodyText2"]),
                Paragraph(f"{late:.1f}", styles["BodyText2"]),
                Paragraph(change_str, styles["BodyText2"]),
            ])

        change_table = Table(change_rows, colWidths=[150, 90, 90, 90])
        change_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(change_table)
        elements.append(Spacer(1, 10))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 3: MANGROVE ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(PageBreak())
    elements.append(Paragraph("3. Mangrove & Coastal Analysis", styles["SectionHeader"]))

    mang = land.get("mangrove", {})
    mang_rows = [
        ["Current Mangrove Presence", "Yes" if mang.get("present") else "No"],
        ["Current Mangrove Area", f"{mang.get('area_ha', 0):.1f} ha"],
        ["Historical Mangrove Detected", "Yes" if mang.get("historical") else "No"],
        ["Mangrove Loss", f"{mang.get('loss_ha', 0):.1f} ha"],
        ["Tidal Wetland Conditions", "Yes" if mang.get("tidal_wetland") else "No"],
    ]

    mang_table = Table(
        [[Paragraph(r[0], styles["BodyText2"]),
          Paragraph(str(r[1]), styles["BodyText2"])] for r in mang_rows],
        colWidths=[250, 200],
    )
    mang_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), BRAND_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(mang_table)
    elements.append(Spacer(1, 10))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 4: ENVIRONMENTAL CONTEXT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(Paragraph("4. Environmental & Climate Context", styles["SectionHeader"]))

    env = land.get("environment", {})
    env_rows = [
        ["Climate Zone", env.get("climate_zone", "N/A")],
        ["Annual Rainfall", f"{env.get('rainfall_mm', 0):.0f} mm"],
        ["Mean Elevation", f"{env.get('elevation_m', 0):.0f} m"],
        ["Mean Slope", f"{env.get('slope_deg', 0):.1f}°"],
        ["Mean NDVI", f"{env.get('mean_ndvi', 0):.3f}"],
        ["Soil Organic Carbon", f"{env.get('soil_carbon_g_per_kg', 0):.1f} g/kg"],
        ["Protected Area Overlap", "Yes" if env.get("in_protected_area") else "No"],
    ]

    env_table = Table(
        [[Paragraph(r[0], styles["BodyText2"]),
          Paragraph(str(r[1]), styles["BodyText2"])] for r in env_rows],
        colWidths=[250, 200],
    )
    env_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), BRAND_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(env_table)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 5: ELIGIBILITY RESULTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(PageBreak())
    elements.append(Paragraph("5. Gold Standard Eligibility Assessment", styles["SectionHeader"]))

    # Eligibility comparison chart
    if results_list:
        elig_path = os.path.join(chart_dir, "eligibility_chart.png")
        generate_eligibility_chart(results_list, elig_path)
        elements.append(Image(elig_path, width=450, height=230))
        elements.append(Spacer(1, 15))

    for idx, r in enumerate(results_list):
        project_type = r.get("project_type", "")
        status = r.get("status", "")
        confidence = r.get("confidence", 0)

        # Status color
        status_style = "StatusGood" if status in ["Highly Suitable", "Suitable"] else "StatusBad"

        elements.append(Paragraph(
            f"5.{idx+1} {project_type}",
            styles["SubSection"]
        ))

        # Status badge
        elements.append(Paragraph(
            f"Status: {status} | Confidence: {confidence:.0f}%",
            styles[status_style]
        ))
        elements.append(Paragraph(
            f"Methodology: {r.get('methodology', 'N/A')}",
            styles["SmallText"]
        ))
        elements.append(Spacer(1, 4))

        # Requirements met
        reqs = r.get("gs_requirements_met", {})
        if reqs:
            req_rows = [[
                Paragraph("<b>Requirement</b>", styles["BodyText2"]),
                Paragraph("<b>Met?</b>", styles["BodyText2"]),
            ]]
            for req_name, met in reqs.items():
                req_rows.append([
                    Paragraph(req_name, styles["BodyText2"]),
                    Paragraph("✓ Yes" if met else "✗ No", styles["BodyText2"]),
                ])

            req_table = Table(req_rows, colWidths=[350, 80])
            req_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(req_table)
            elements.append(Spacer(1, 4))

        # Reasons
        for reason in r.get("reasons", []):
            elements.append(Paragraph(f"• {reason}", styles["BodyText2"]))

        # Warnings
        for warning in r.get("warnings", []):
            elements.append(Paragraph(f"⚠ {warning}", styles["Warning"]))

        elements.append(Spacer(1, 10))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 6: RECOMMENDATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(PageBreak())
    elements.append(Paragraph("6. Recommendation & Next Steps", styles["SectionHeader"]))

    narrative = eligibility_results.get("narrative", "")
    # Convert markdown-like formatting
    for line in narrative.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("**") and line.endswith("**"):
            elements.append(Paragraph(
                line.replace("**", ""),
                styles["SubSection"]
            ))
        elif line.startswith("- "):
            elements.append(Paragraph(f"• {line[2:]}", styles["BodyText2"]))
        elif line.startswith(("1.", "2.", "3.", "4.", "5.")):
            elements.append(Paragraph(line, styles["BodyText2"]))
        else:
            clean = line.replace("**", "<b>").replace("**", "</b>")
            # Simple bold conversion
            import re
            clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            elements.append(Paragraph(clean, styles["BodyText2"]))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FOOTER / DISCLAIMER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    elements.append(Paragraph(
        "<b>Disclaimer:</b> This pre-feasibility report is generated using satellite-derived data "
        "and automated spatial analysis. Results are indicative and must be validated through "
        "field verification, detailed Project Design Document (PDD) preparation, and formal "
        "Gold Standard validation. Land use classifications are based on 10m resolution "
        "satellite imagery and may not capture fine-scale features.",
        styles["SmallText"]
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Generated by Flora Carbon AI Pre-Feasibility Screening Tool | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["SmallText"]
    ))

    # Build PDF
    doc.build(elements)

    # Clean up temp chart files
    for f in ["lulc_pie_latest.png", "forest_loss_chart.png", "ndvi_chart.png", "eligibility_chart.png"]:
        p = os.path.join(chart_dir, f)
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass

    return output_path
