"""
app.py — Gradio UI for Carbon Emission Pipeline
Deploy on Hugging Face Spaces (Gradio SDK)
"""

import gradio as gr
import json
import os
import re
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline.orchestrator import run_full_pipeline
from utils.storage import get_trend, save_run
from services.insights_store import get_dashboard_snapshot
from services.email_service import send_report_email, should_send_email
from services.health_check import update_pipeline_health, get_status_color_and_icon
# ─── Chart Builders ───────────────────────────────────────────────────────────

def build_scope_chart(totals: dict) -> go.Figure:
    scopes = ["Scope 1\n(Direct)", "Scope 2\n(Energy)", "Scope 3\n(Supply Chain)"]
    values = [
        totals.get("scope1_kg", 0),
        totals.get("scope2_kg", 0),
        totals.get("scope3_kg", 0)
    ]
    colors = ["#ef4444", "#f97316", "#3b82f6"]
    fig = go.Figure(go.Bar(
        x=scopes, y=values, marker_color=colors,
        text=[f"{v:.1f} kg" for v in values],
        textposition="outside"
    ))
    fig.update_layout(
        title="Emissions by Scope (kg CO2e)",
        yaxis_title="kg CO2e",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        showlegend=False,
        height=350
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.16)", zeroline=False)
    return fig


def build_activity_pie(by_type: dict) -> go.Figure:
    labels = list(by_type.keys())
    values = list(by_type.values())
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker_colors=["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
    ))
    fig.update_layout(
        title="Emissions by Activity Type",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        height=350
    )
    return fig


def build_recommendations_chart(recommendations: list) -> go.Figure:
    if not recommendations:
        return go.Figure()
    recs = recommendations[:6]
    titles = [r["title"][:30] + "..." if len(r["title"]) > 30 else r["title"] for r in recs]
    savings = [r.get("co2e_savings_kg", 0) for r in recs]
    scores = [r.get("priority_score", 0) for r in recs]

    fig = go.Figure(go.Bar(
        y=titles, x=savings,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Priority")
        ),
        text=[f"{s:.1f} kg saved" for s in savings],
        textposition="outside"
    ))
    fig.update_layout(
        title="Reduction Opportunities (kg CO2e savings)",
        xaxis_title="Potential Savings (kg CO2e)",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        margin=dict(l=20, r=30, t=50, b=40),
        height=400
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Priority"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.16)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


def format_compact_value(value: float) -> str:
    """Format values in a compact, readable way."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    if abs_value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def format_kpi_cards(totals: dict, pot_savings: float) -> str:
    """Render the top summary as modern KPI cards instead of a table."""
    total_kg = totals.get("total_kg", 0)
    scope1 = totals.get("scope1_kg", 0)
    scope3 = totals.get("scope3_kg", 0)

    cards = [
        ("Total CO2e", f"{format_compact_value(total_kg)} kg", "Total footprint from the uploaded document"),
        ("Scope 1", f"{format_compact_value(scope1)} kg", "Direct emissions"),
        ("Scope 3", f"{format_compact_value(scope3)} kg", "Supply chain emissions"),
        ("Savings", f"{format_compact_value(pot_savings)} kg", "Highest reduction opportunity"),
    ]

    card_html = []
    for title, value, caption in cards:
        card_html.append(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-caption">{caption}</div>
            </div>
            """
        )

    return f"""
    <div class="kpi-grid">
        {''.join(card_html)}
    </div>
    """


def format_extracted_items(items: list) -> str:
    if not items:
        return "## Extracted Items\n\nNo extracted items found."

    lines = ["## Extracted Items", ""]
    for index, item in enumerate(items[:8], 1):
        lines.append(f"### Item {index}")
        lines.append(f"- Description: {item.get('description') or 'N/A'}")
        lines.append(f"- Activity Type: {item.get('activity_type') or 'N/A'}")
        lines.append(f"- Activity Subtype: {item.get('activity_subtype') or 'N/A'}")
        lines.append(f"- Quantity: {item.get('quantity') if item.get('quantity') is not None else 'N/A'}")
        lines.append(f"- Unit: {item.get('unit') or 'N/A'}")
        lines.append(f"- Transport Mode: {item.get('transport_mode') or 'N/A'}")
        lines.append(f"- Origin → Destination: {(item.get('origin') or 'N/A')} → {(item.get('destination') or 'N/A')}")
        lines.append(f"- Distance: {item.get('distance') if item.get('distance') is not None else 'N/A'} {item.get('distance_unit') or ''}".strip())
        lines.append(f"- Confidence: {item.get('confidence') or 'N/A'}")
        lines.append("")

    if len(items) > 8:
        lines.append(f"_Showing first 8 of {len(items)} extracted items._")

    return "\n".join(lines)


def format_narrative_insights(
    totals: dict,
    by_type: dict,
    recommendations: list,
    extracted_items: list,
    emission_validation: dict | None = None,
) -> str:
    """Create a plain-language interpretation for non-technical users."""
    total_kg = totals.get("total_kg", 0)
    scope1 = totals.get("scope1_kg", 0)
    scope2 = totals.get("scope2_kg", 0)
    scope3 = totals.get("scope3_kg", 0)

    def share(value: float) -> float:
        return (value / total_kg * 100) if total_kg else 0

    dominant_scope_name = "Scope 3 (Supply Chain)"
    dominant_scope_value = scope3
    if scope1 >= scope2 and scope1 >= scope3:
        dominant_scope_name = "Scope 1 (Direct)"
        dominant_scope_value = scope1
    elif scope2 >= scope1 and scope2 >= scope3:
        dominant_scope_name = "Scope 2 (Energy)"
        dominant_scope_value = scope2

    dominant_activity = None
    dominant_activity_value = 0
    if by_type:
        dominant_activity, dominant_activity_value = max(by_type.items(), key=lambda item: item[1])

    top_rec = max(recommendations, key=lambda r: r.get("co2e_savings_kg", 0), default=None)
    items_count = len(extracted_items)

    lines = ["## Executive Narrative", ""]
    lines.append("### What happened")
    if total_kg > 0:
        lines.append(
            f"Your footprint is {total_kg:.2f} kg CO2e. {dominant_scope_name} is the largest source at "
            f"{dominant_scope_value:.2f} kg CO2e ({share(dominant_scope_value):.1f}% of total)."
        )
        if scope3 > 0 and scope1 > 0:
            ratio = scope3 / scope1 if scope1 else 0
            if ratio >= 10:
                lines.append(f"Scope 3 is about {ratio:.1f}x higher than Scope 1, so most emissions are coming from suppliers and purchased goods.")
            elif ratio >= 3:
                lines.append(f"Scope 3 is materially higher than Scope 1, which means external sourcing is the main carbon driver.")
    else:
        lines.append("No measurable emissions were calculated from the uploaded file.")

    if dominant_activity:
        lines.append("")
        lines.append("### Why it matters")
        lines.append(
            f"The main activity type is {dominant_activity}, contributing {dominant_activity_value:.2f} kg CO2e "
            f"({share(dominant_activity_value):.1f}% of total). That means the biggest improvement will come from "
            f"targeting this area first instead of spreading effort evenly across the business."
        )
    else:
        lines.append("")
        lines.append("### Why it matters")
        lines.append(
            "The file did not provide enough structured activity information to determine the main emissions driver clearly. "
            "Reviewing supplier lines and quantity units will improve the result."
        )

    lines.append("")
    lines.append("### What to do next")
    if top_rec:
        lines.append(
            f"Prioritize: {top_rec.get('title', 'the highest-savings recommendation')} "
            f"({top_rec.get('co2e_savings_kg', 0):.1f} kg CO2e potential savings)."
        )
        lines.append(
            f"This is the fastest way to reduce emissions because it targets the largest available savings first."
        )
    else:
        lines.append(
            "Start with supplier data cleanup, then rerun the analysis so the system can identify the largest saving opportunity."
        )

    if items_count > 0:
        lines.append("")
        lines.append(f"The analysis is based on {items_count} extracted line items.")

    if emission_validation:
        coverage = emission_validation.get("coverage", {})
        deviation = emission_validation.get("deviation_percent", emission_validation.get("comparison", {}).get("effective_diff_pct", "N/A"))
        confidence_label = emission_validation.get("confidence", "UNKNOWN")
        confidence_score = emission_validation.get("confidence_score_pct", 0)
        lines.append("")
        lines.append("### Validation")
        lines.append(
            f"Status: {emission_validation.get('status', 'N/A')} | "
            f"Coverage: {coverage.get('mapped_items', 0)}/{coverage.get('total_items', 0)} "
            f"({coverage.get('coverage_pct', 0)}%) | "
            f"Diff: {deviation}% | "
            f"Confidence: {confidence_score}% ({confidence_label})."
        )

        why = emission_validation.get("why_difference", [])
        if why:
            lines.append("Potential difference drivers: " + "; ".join(why[:3]) + ".")

    return "\n".join(lines)


def format_validation_report_html(emission_validation: dict) -> str:
        if not emission_validation:
                return "<div class='summary-note'>Validation data unavailable.</div>"

        status = emission_validation.get("status", "REVIEW")
        confidence = emission_validation.get("confidence", "UNKNOWN")
        deviation = emission_validation.get("deviation_percent")
        coverage = emission_validation.get("coverage", {})
        system_total = emission_validation.get("system_total", 0)
        reference_total = emission_validation.get("reference_total", 0)
        explanation = emission_validation.get("explanation", "No explanation available.")
        mode = emission_validation.get("mode", "auto")
        breakdown = emission_validation.get("breakdown", {})

        if status == "APPROVED":
                status_color = "#10b981"
                status_badge = "✔ APPROVED"
        elif status == "REVIEW":
                status_color = "#f59e0b"
                status_badge = "⚠ REVIEW"
        else:
                status_color = "#ef4444"
                status_badge = "✖ REJECTED"

        def confidence_color(diff):
                if diff is None:
                        return "#94a3b8"
                if diff <= 5:
                        return "#10b981"
                if diff <= 10:
                        return "#f59e0b"
                return "#ef4444"

        def scope_row(scope_key: str, title: str) -> str:
                scope = breakdown.get(scope_key, {})
                s = scope.get("system")
                r = scope.get("reference")
                d = scope.get("deviation_percent")
                c = scope.get("confidence", "UNKNOWN")
                bar_color = confidence_color(d)
                width = 0 if d is None else min(d, 100)
                return (
                        f"<tr>"
                        f"<td>{title}</td>"
                        f"<td>{s if s is not None else 'N/A'}</td>"
                        f"<td>{r if r is not None else 'N/A'}</td>"
                        f"<td>{d if d is not None else 'N/A'}%</td>"
                        f"<td>{c}</td>"
                        f"<td><div style='background: rgba(255,255,255,0.08); height: 8px; border-radius: 6px; overflow: hidden; min-width: 120px;'>"
                        f"<div style='width:{width}%; background:{bar_color}; height:100%;'></div></div></td>"
                        f"</tr>"
                )

        return f"""
        <div class='summary-note' style='border-left-color:{status_color};'>
            <h3 style='margin:0 0 8px 0;'>Validation Report</h3>
            <div><b>Mode:</b> {mode}</div>
            <div><b>Status:</b> <span style='color:{status_color};font-weight:700'>{status_badge}</span></div>
            <div><b>Deviation:</b> {deviation if deviation is not None else 'N/A'}%</div>
            <div><b>Confidence:</b> {confidence} ({emission_validation.get('confidence_score_pct', 0)}%)</div>
            <div><b>Coverage:</b> {coverage.get('mapped_items', 0)}/{coverage.get('total_items', 0)} ({coverage.get('coverage_pct', 0)}%)</div>
            <div><b>System Total:</b> {system_total} kg CO2e</div>
            <div><b>Reference Total:</b> {reference_total if reference_total is not None else 'N/A'} kg CO2e</div>
            <div style='margin-top:8px;'><b>Explanation:</b> {explanation}</div>
            <div style='margin-top:10px;'>
                <table style='width:100%; border-collapse: collapse;'>
                    <thead>
                        <tr>
                            <th style='text-align:left;'>Scope</th><th>System</th><th>Reference</th><th>Diff</th><th>Confidence</th><th>Bar</th>
                        </tr>
                    </thead>
                    <tbody>
                        {scope_row('scope1', 'Scope 1')}
                        {scope_row('scope2', 'Scope 2')}
                        {scope_row('scope3', 'Scope 3')}
                    </tbody>
                </table>
            </div>
        </div>
        """


def _markdown_to_plain_lines(markdown: str) -> list[str]:
    """Convert markdown report content to clean plain text lines."""
    if not markdown:
        return []

    lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"`(.*?)`", r"\1", line)
        line = line.replace("|", " ")
        lines.append(line)

    return lines


def create_report_pdf(report_markdown: str) -> str | None:
    """Create a temporary PDF file and return its path for download."""
    if not report_markdown:
        return None

    # Normalize any remaining unsupported glyphs before ReportLab renders the PDF.
    report_markdown = (
        report_markdown
        .replace("CO₂e", "CO2e")
        .replace("CO₂", "CO2")
        .replace("₂", "2")
    )

    pdf_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix="_carbon_report.pdf").name)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=(8.27 * inch, 11.69 * inch),
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Carbon Emissions Audit Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111827"),
        alignment=TA_LEFT,
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#7f1d1d"),
        spaceBefore=10,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#111827"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#111827"),
        spaceAfter=5,
    )

    def esc(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    story = [Paragraph("Carbon Emissions Audit Report", title_style), Spacer(1, 8)]
    for line in _markdown_to_plain_lines(report_markdown):
        if not line:
            story.append(Spacer(1, 4))
            continue
        if line.startswith("Executive Summary") or line.startswith("Emissions Overview") or line.startswith("Emissions by Activity Category") or line.startswith("Line-Item Breakdown") or line.startswith("Reduction Recommendations") or line.startswith("Methodology"):
            story.append(Paragraph(esc(line), h1_style))
        elif line.startswith("What happened") or line.startswith("Why it matters") or line.startswith("What to do next"):
            story.append(Paragraph(esc(line), h2_style))
        elif line.startswith("|") and "---" not in line:
            cols = [c.strip() for c in line.strip("|").split("|")]
            table_data = [[Paragraph(esc(c), body_style) for c in cols]]
            table = Table(table_data, colWidths=[2.3 * inch, 3.8 * inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(table)
        elif line.startswith("-"):
            story.append(Paragraph("• " + esc(line[1:].strip()), body_style))
        else:
            story.append(Paragraph(esc(line), body_style))

    try:
        doc.build(story)
        return str(pdf_path)
    except Exception:
        return None


# ─── Main Processing Function ─────────────────────────────────────────────────

def _normalize_company_id(company_name: str | None) -> str:
    if not company_name:
        return "demo_company"
    company_id = company_name.strip().lower()
    company_id = re.sub(r"[^a-z0-9]+", "_", company_id).strip("_")
    return company_id or "demo_company"


def process_document(
    file,
    region,
    company_name,
    email_address=None,
    validation_mode="Auto Analysis",
    manual_scope1=None,
    manual_scope2=None,
    manual_scope3=None,
    manual_total=None,
):
    if file is None:
        return (
            "Please upload a file first.",
            "",
            "",
            None, None, None,
            "No data",
            "No data",
            "",
            None
        )

    region_map = {"US (EPA)": "us", "UK (DEFRA)": "uk", "India": "in"}
    region_code = region_map.get(region, "us")

    manual_validation = None
    if validation_mode == "Validate Results":
        manual_validation = {
            "manual_scope1": manual_scope1,
            "manual_scope2": manual_scope2,
            "manual_scope3": manual_scope3,
            "manual_total": manual_total,
        }

    try:
        file_path = file.name if hasattr(file, "name") else str(file)
        result = run_full_pipeline(file_path, region=region_code, manual_validation=manual_validation)

        if not result.get("success"):
            return (
                f"Pipeline error: {result.get('error', 'Unknown error')}",
                "",
                "",
                None, None, None,
                "Error",
                "Error",
                "",
                None
            )

        po = result["pipeline_output"]
        analyst = po.get("analyst", {})
        recommender = po.get("recommender", {})
        emission_validation = po.get("emission_validation", {})
        report = po.get("report", {})
        extracted = po.get("extracted", {})
        totals = analyst.get("totals", {})
        by_type = analyst.get("by_activity_type", {})
        recs = recommender.get("recommendations", [])
        extracted_items = po.get("extracted", {}).get("data", {}).get("items", [])

        # Summary card
        total_kg = totals.get("total_kg", 0)
        pot_savings = recommender.get("total_potential_savings_kg", 0)

        summary_html = format_kpi_cards(totals, pot_savings)

        company_id = _normalize_company_id(company_name)
        full_report = report.get("markdown_report", "Report generation failed")
        save_run(company_id, result.get("summary", {}), analyst, full_report)
        trend = get_trend(company_id)

        if trend.get("trend") == "insufficient_data":
            summary_html += (
                f"<div class='summary-note'>Company: <b>{company_id}</b><br/>"
                f"Historical trend will appear after additional analysis runs. "
                f"Current runs: {trend.get('runs', 0)}.</div>"
            )
        else:
            direction = "increase" if trend.get("trend") == "up" else "decrease"
            summary_html += (
                f"<div class='summary-note'>Company: <b>{company_id}</b><br/>"
                f"Trend vs last run: <b>{trend.get('delta_pct', 0):+.1f}%</b> "
                f"({trend.get('delta_kg', 0):+.2f} kg CO2e, {direction}).</div>"
            )

        if extracted.get("fallback_mode") and extracted.get("warning"):
            summary_html += (
                "<div class='summary-note'>Running in fallback extraction mode. "
                "Set GROQ_API_KEY for higher-quality LLM extraction.</div>"
            )

        narrative_md = format_narrative_insights(
            totals,
            by_type,
            recs,
            extracted_items,
            emission_validation=emission_validation,
        )
        validation_html = format_validation_report_html(emission_validation)

        # Charts
        scope_chart = build_scope_chart(totals) if total_kg > 0 else None
        pie_chart = build_activity_pie(by_type) if by_type else None
        rec_chart = build_recommendations_chart(recs)

        # Extraction formatted preview
        extraction_text = format_extracted_items(extracted_items)

        # Recommendations markdown
        rec_md_lines = ["## Top Reduction Opportunities\n"]
        for i, r in enumerate(recs[:5], 1):
            rec_md_lines.append(
                f"**{i}. {r.get('title')}** — Priority {r.get('priority_score')}/10  \n"
                f"{r.get('description', '')}  \n"
                    f"Saves **{r.get('co2e_savings_kg', 0):.1f} kg CO2e** | "
                f"Effort: {r.get('implementation_effort', '').title()} | "
                f"Timeline: {r.get('timeframe', '').replace('_', ' ').title()}\n"
            )
        rec_md = "\n".join(rec_md_lines)

        # Full markdown report
        report_pdf_file = create_report_pdf(full_report)
        
        # Send email if configured and email provided
        email_status = ""
        if email_address and report_pdf_file:
            email_result = send_report_email(
                pdf_path=report_pdf_file,
                recipient_email=email_address,
                company_name=company_id,
                total_emissions_kg=total_kg
            )
            if email_result.get("success"):
                email_status = f"<div class='summary-note' style='border-left-color: #10b981; background: rgba(16, 185, 129, 0.12); color: #86efac;'><b>Email sent:</b> {email_result.get('message')}</div>"
            else:
                email_status = f"<div class='summary-note'><b>Email failed:</b> {email_result.get('message')}</div>"
            summary_html += email_status

        return (
            summary_html,
            narrative_md,
            validation_html,
            scope_chart,
            pie_chart,
            rec_chart,
            extraction_text,
            rec_md,
            full_report,
            report_pdf_file
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return (
            f"Unexpected error: {str(e)}",
            "",
            "",
            None, None, None,
            tb[:500],
            "Error",
            "",
            None
        )


def get_kafka_pipeline_status() -> str:
    """Fetch and format real Kafka pipeline status from insights store with health checks."""
    try:
        snapshot = get_dashboard_snapshot()
        
        # Update health status based on actual Kafka connectivity
        snapshot = update_pipeline_health(snapshot)
        
        pipeline = snapshot.get("pipeline", {})
        emissions = snapshot.get("emissions", {})
        scopes = emissions.get("scope_distribution", {})
        
        # Get status details
        status = pipeline.get("status", "unknown")
        status_reason = pipeline.get("status_reason", "")
        status_color, status_icon, status_label = get_status_color_and_icon(status)
        
        last_event = pipeline.get("last_event_received")
        if last_event:
            from datetime import datetime as dt
            try:
                event_time = dt.fromisoformat(last_event.replace("Z", "+00:00"))
                now = dt.now(event_time.tzinfo)
                seconds_ago = (now - event_time).total_seconds()
                recency = f"{int(seconds_ago)}s ago" if seconds_ago < 60 else f"{int(seconds_ago/60)}m ago"
            except:
                recency = "unknown"
        else:
            recency = "no events"
        
        # Build status indicator with color coding
        status_indicator = f"<span style='color: {status_color}; font-weight: bold;'>{status_icon} {status_label}</span>"
        
        # Determine section visibility based on status
        metrics_section = ""
        if status != "offline":
            metrics_section = f"""
**Processing Metrics**
- Events Processed: **{pipeline.get("events_processed_total", 0):,}**
- Session Throughput: **{pipeline.get("kafka_throughput_per_min", 0):.1f}** events/min
- Consumer Latency: **{pipeline.get("consumer_latency_ms", 0):.1f}** ms
- Queue Depth: **{pipeline.get("processing_queue_depth", 0)}** messages
"""
        else:
            metrics_section = """
**Processing Metrics**
- Status: **Kafka broker is offline** — no metrics available
- Last known throughput: Check when Kafka is back online
"""
        
        status_md = f"""
### Enterprise Carbon Pipeline Status

**Pipeline Health**
- Status: {status_indicator}
- Last Event: **{recency}**
- Reason: {status_reason}
- Kafka Topic: `carbon-events`
- Consumer Group: `carbon-processor-group`
{metrics_section}

**Emissions Aggregates** (cached from latest events)
- Total CO2e Session: **{emissions.get("total_kg_session", 0):,.2f}** kg
- Scope 1 (Direct): **{emissions.get("scope1_kg", 0):,.2f}** kg ({scopes.get("scope1", 0)}%)
- Scope 2 (Energy): **{emissions.get("scope2_kg", 0):,.2f}** kg ({scopes.get("scope2", 0)}%)
- Scope 3 (Supply Chain): **{emissions.get("scope3_kg", 0):,.2f}** kg ({scopes.get("scope3", 0)}%)

**Data Quality**
- Suppliers Connected: **{snapshot.get("activity", {}).get("supplier_count", 0)}**
- Regions: **{snapshot.get("activity", {}).get("region_count", 0)}**
- Confidence Score: **{snapshot.get("activity", {}).get("confidence_avg", 0):.1%}**
"""
        
        # Show interpretation only if pipeline is connected
        interpretation = snapshot.get("ai_insights", {}).get("interpretation", {}).get("primary", "")
        if status == "connected" and interpretation:
            status_md += f"""

**Live Interpretation**
{interpretation}"""
        elif status == "offline":
            status_md += f"""

**Action Required**
Start Kafka and consumer to resume event processing:
```bash
# Start infrastructure
docker-compose up -d

# Start services  
./startup.sh start
```"""
        else:
            status_md += """

**Status**
Pipeline is stale — waiting for new events from connected sources."""
        
        status_md += """

---

*This pipeline connects enterprise data sources (ERP, IoT sensors, billing systems, logistics feeds) to the carbon intelligence engine.*
"""

        return status_md
    except Exception as e:
        return f"Pipeline status unavailable: {str(e)}"


def refresh_kafka_status():
    """Refresh Kafka pipeline status display."""
    return get_kafka_pipeline_status()

# ─── Gradio UI ─────────────────────────────────────────────────────────────────

NAV_HERO_HTML = """
<div class="hero-wrap">
    <div class="top-nav">
        <div class="brand">
            <span class="brand-mark">CI</span>
            <span>Carbon Intelligence</span>
        </div>
        <div class="nav-meta">
            <span class="meta-dot"></span>
            <span>Carbon Operations Console</span>
            <span class="meta-divider">•</span>
            <span>Live Pipeline</span>
        </div>
        <div class="nav-cta">Enterprise ESG</div>
    </div>

    <div class="hero-content">
        <h1>Rise Above Emissions Noise.<br/>Audit With Precision.</h1>
        <p>
            Carbon workflow platform for invoices, manifests, and energy bills.
            Extract data, calculate CO2e, and generate ESG-ready insights in one flow.
        </p>

        <div class="workflow-steps" role="list" aria-label="Pipeline steps">
            <div class="step-item" role="listitem"><span class="step-no">01</span><span class="step-label">Extract</span></div>
            <div class="step-item" role="listitem"><span class="step-no">02</span><span class="step-label">Validate</span></div>
            <div class="step-item" role="listitem"><span class="step-no">03</span><span class="step-label">Analyze</span></div>
            <div class="step-item" role="listitem"><span class="step-no">04</span><span class="step-label">Recommend</span></div>
            <div class="step-item" role="listitem"><span class="step-no">05</span><span class="step-label">Report</span></div>
        </div>
    </div>
</div>
"""

APP_CSS = """
    .gradio-container {
        background: radial-gradient(1000px 420px at 80% -10%, #3f1115 0%, #121214 45%, #0a0a0b 100%);
        min-height: 100vh;
    }

    .app-shell {
        max-width: 1240px;
        margin: 0 auto;
        padding: 12px 20px 24px;
    }

    .hero-wrap {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        background: linear-gradient(125deg, rgba(20, 20, 22, 0.96), rgba(27, 16, 18, 0.95));
        backdrop-filter: blur(4px);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.5);
        padding: 16px 24px 20px;
        margin-bottom: 14px;
    }

    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 18px;
        font-size: 0.95rem;
        color: #f3f4f6;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        letter-spacing: 0.01em;
        white-space: nowrap;
    }

    .brand-mark {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        border: 1px solid rgba(239, 68, 68, 0.55);
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.25), rgba(127, 29, 29, 0.2));
    }

    .nav-meta {
        display: flex;
        align-items: center;
        gap: 10px;
        opacity: 0.88;
        flex-wrap: wrap;
        justify-content: center;
        font-weight: 500;
        color: #d4d4d8;
        font-size: 0.9rem;
    }

    .meta-dot {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: #ef4444;
        box-shadow: 0 0 0 5px rgba(239, 68, 68, 0.2);
        display: inline-block;
    }

    .meta-divider {
        opacity: 0.6;
    }

    .nav-cta {
        border: 1px solid rgba(239, 68, 68, 0.52);
        border-radius: 999px;
        padding: 8px 14px;
        background: rgba(127, 29, 29, 0.2);
        white-space: nowrap;
        font-weight: 600;
    }

    .hero-content h1 {
        margin: 0 0 10px;
        font-size: clamp(2rem, 3.7vw, 2.9rem);
        line-height: 1.04;
        color: #f5f5f5;
        letter-spacing: -0.02em;
    }

    .hero-content p {
        margin: 0;
        max-width: 720px;
        color: #d4d4d8;
        font-size: 0.98rem;
        line-height: 1.5;
    }

    .workflow-steps {
        margin-top: 20px;
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
    }

    .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        background: rgba(24, 24, 27, 0.72);
        border-radius: 10px;
        padding: 9px 10px;
    }

    .step-no {
        width: 28px;
        height: 28px;
        border-radius: 7px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.76rem;
        font-weight: 700;
        color: #fecaca;
        border: 1px solid rgba(239, 68, 68, 0.52);
        background: rgba(185, 28, 28, 0.24);
    }

    .step-label {
        color: #f4f4f5;
        font-size: 0.92rem;
        font-weight: 600;
    }

    .panel {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(14, 14, 16, 0.84);
        padding: 18px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
    }

    .summary-box {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(14, 14, 16, 0.86);
        padding: 16px 18px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
        min-height: 100%;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 10px;
    }

    .kpi-card {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: linear-gradient(180deg, rgba(24, 24, 27, 0.95), rgba(17, 17, 19, 0.92));
        box-shadow: 0 14px 28px rgba(0, 0, 0, 0.22);
        padding: 14px 16px;
        min-height: 118px;
    }

    .kpi-title {
        color: #cbd5e1;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: #fff7ed;
        font-size: clamp(1.5rem, 2.4vw, 2rem);
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 8px;
    }

    .kpi-caption {
        color: #d4d4d8;
        font-size: 0.9rem;
        line-height: 1.35;
    }

    .summary-note {
        margin-top: 12px;
        padding: 10px 12px;
        border-left: 3px solid #ef4444;
        background: rgba(127, 29, 29, 0.12);
        color: #fecaca;
        border-radius: 10px;
        font-size: 0.92rem;
    }

    .gr-tabs {
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 14px;
        background: rgba(14, 14, 16, 0.82);
        padding: 12px;
    }

    .chart-card {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(16, 16, 18, 0.9);
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.22);
        padding: 10px 10px 4px;
        overflow: hidden;
    }

    .content-panel {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(16, 16, 18, 0.88);
        padding: 14px 16px;
        box-shadow: 0 14px 28px rgba(0, 0, 0, 0.2);
    }

    .section-heading {
        color: #f5f5f5;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 0 10px;
    }

    .body-copy {
        color: #d4d4d8;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .caption-copy {
        color: #a1a1aa;
        font-size: 0.84rem;
        line-height: 1.4;
    }

    .gr-tabs .prose,
    .gr-tabs .prose p,
    .gr-tabs .prose li,
    .gr-tabs .prose strong {
        color: #e5e7eb;
    }

    .gr-tabs .prose h1,
    .gr-tabs .prose h2,
    .gr-tabs .prose h3,
    .gr-tabs .prose h4 {
        color: #fee2e2;
    }

    .gr-tabs table {
        width: 100%;
        border-collapse: collapse;
        background: rgba(10, 10, 11, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .gr-tabs th {
        background: rgba(127, 29, 29, 0.45);
        color: #fecaca;
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 8px 10px;
    }

    .gr-tabs td {
        color: #e5e7eb;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 8px 10px;
    }

    .gr-tabs pre,
    .gr-tabs code {
        background: rgba(10, 10, 11, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #fca5a5 !important;
    }

    @media (max-width: 920px) {
        .top-nav {
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
        }

        .nav-meta {
            justify-content: flex-start;
            gap: 12px;
        }

        .workflow-steps {
            grid-template-columns: 1fr;
        }

        .kpi-grid {
            grid-template-columns: 1fr 1fr;
        }
    }

    @media (max-width: 640px) {
        .kpi-grid {
            grid-template-columns: 1fr;
        }
    }

    footer { display: none !important; }
    """


with gr.Blocks(title="Carbon Emission Analyzer") as demo:

    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(NAV_HERO_HTML)

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                company_input = gr.Textbox(
                    label="Company Name",
                    placeholder="e.g. Vertex Industries",
                    value="demo_company"
                )
                file_input = gr.File(
                    label="Upload Document (PDF or CSV)",
                    file_types=[".pdf", ".csv", ".txt"],
                    type="filepath"
                )
                region_select = gr.Radio(
                    choices=["US (EPA)", "UK (DEFRA)", "India"],
                    value="US (EPA)",
                    label="Region (for emission factors)"
                )
                email_input = gr.Textbox(
                    label="Email Address (optional)",
                    placeholder="Enter email to receive PDF report",
                    type="email",
                    value=""
                )

                validation_mode = gr.Radio(
                    choices=["Auto Analysis", "Validate Results"],
                    value="Auto Analysis",
                    label="Validation Mode"
                )
                with gr.Column(visible=False) as validation_panel:
                    gr.Markdown("### Manual Verification (Optional)")
                    manual_scope1_input = gr.Number(label="Manual Scope 1 (kg CO2)", value=None)
                    manual_scope2_input = gr.Number(label="Manual Scope 2 (kg CO2)", value=None)
                    manual_scope3_input = gr.Number(label="Manual Scope 3 (kg CO2)", value=None)
                    manual_total_input = gr.Number(label="Manual Total (kg CO2, optional)", value=None)

                analyze_btn = gr.Button("Analyze Emissions", variant="primary", size="lg")

                gr.Markdown("### Try Sample Documents")
                gr.Markdown(
                    "Download and upload these to test:\n"
                    "- `sample_docs/sample_invoice.csv`\n"
                    "- `sample_docs/sample_manifest.csv`"
                )

            with gr.Column(scale=2):
                summary_out = gr.HTML(
                    "Upload a document and click Analyze Emissions to begin.",
                    elem_classes=["summary-box", "content-panel"]
                )

        narrative_out = gr.Markdown(elem_classes=["summary-box", "content-panel"])
        validation_out = gr.HTML(elem_classes=["summary-box", "content-panel"])

        with gr.Tabs(elem_classes=["gr-tabs"]):
            with gr.TabItem("Visualizations"):
                with gr.Row():
                    scope_chart_out = gr.Plot(label="Scope Breakdown", elem_classes=["chart-card"])
                    pie_chart_out = gr.Plot(label="Activity Breakdown", elem_classes=["chart-card"])
                rec_chart_out = gr.Plot(label="Reduction Opportunities", elem_classes=["chart-card"])

            with gr.TabItem("Event Monitor"):
                pipeline_status = gr.Markdown(get_kafka_pipeline_status())
                refresh_btn = gr.Button("Refresh Status", variant="secondary")
                auto_timer = gr.Timer(5)  # Auto-refresh every 5 seconds
                
                refresh_btn.click(refresh_kafka_status, outputs=pipeline_status)
                auto_timer.tick(refresh_kafka_status, outputs=pipeline_status)

            with gr.TabItem("Extracted Data"):
                extraction_out = gr.Markdown("No extracted data yet.", elem_classes=["content-panel"])

            with gr.TabItem("Recommendations"):
                rec_md_out = gr.Markdown(elem_classes=["content-panel"])

            with gr.TabItem("Full ESG Report"):
                report_out = gr.Markdown(elem_classes=["content-panel"])
                download_report_pdf_btn = gr.DownloadButton(
                    "Download PDF",
                    value=None,
                    elem_classes=["content-panel"]
                )

    analyze_btn.click(
        fn=process_document,
        inputs=[
            file_input,
            region_select,
            company_input,
            email_input,
            validation_mode,
            manual_scope1_input,
            manual_scope2_input,
            manual_scope3_input,
            manual_total_input,
        ],
        outputs=[
            summary_out,
            narrative_out,
            validation_out,
            scope_chart_out,
            pie_chart_out,
            rec_chart_out,
            extraction_out,
            rec_md_out,
            report_out,
            download_report_pdf_btn
        ],
        show_progress="full"
    )

    def _toggle_validation_panel(mode):
        return gr.update(visible=(mode == "Validate Results"))

    validation_mode.change(
        fn=_toggle_validation_panel,
        inputs=[validation_mode],
        outputs=[validation_panel],
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, css=APP_CSS)
