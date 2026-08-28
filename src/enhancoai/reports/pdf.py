"""PDF report generation via ReportLab (no external binary dependency)."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

from enhancoai.reports import ReportData

_TAG_RE = re.compile(r"<[^>]+>")


def render_pdf_report(data: ReportData, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=LETTER)
    story = [Paragraph(f"{data.project_name} Report", styles["Title"]), Paragraph(f"Generated: {data.generated_at}", styles["Normal"]), Spacer(1, 0.2 * inch)]

    for i, section in enumerate(data.sections, start=1):
        story.append(Paragraph(f"{i}. {section.title}", styles["Heading2"]))
        plain_text = _TAG_RE.sub("", section.html_body).strip()
        if plain_text:
            story.append(Paragraph(plain_text, styles["Normal"]))
        for fig_path in section.figure_paths:
            if Path(fig_path).exists():
                story.append(Spacer(1, 0.1 * inch))
                story.append(Image(fig_path, width=5 * inch, height=3.5 * inch, kind="proportional"))
        story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(
        "Note: computational proxies, not experimental measurements. See the Limitations section.",
        styles["Italic"],
    ))

    doc.build(story)
    return output_path
