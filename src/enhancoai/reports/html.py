"""HTML report generation via Jinja2 (section 48)."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from enhancoai.reports import ReportData

_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ data.project_name }} Report</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }
  h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 0.5rem; }
  h2 { color: #2c3e50; margin-top: 2.5rem; border-left: 4px solid #2c3e50; padding-left: 0.5rem; }
  .meta { color: #666; font-size: 0.9rem; }
  .figure { margin: 1rem 0; max-width: 100%; }
  .figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
  .disclaimer { background: #fff8e1; border: 1px solid #f0c14b; padding: 1rem; border-radius: 4px; margin-top: 2rem; }
</style>
</head>
<body>
<h1>{{ data.project_name }} Report</h1>
<p class="meta">Generated: {{ data.generated_at }}</p>

{% for section in data.sections %}
<h2>{{ loop.index }}. {{ section.title }}</h2>
<div>{{ section.html_body | safe }}</div>
{% for fig in section.figure_paths %}
<div class="figure"><img src="{{ fig }}" alt="{{ section.title }}"></div>
{% endfor %}
{% endfor %}

<div class="disclaimer">
<strong>Note:</strong> This report was generated automatically by EnhancoAI. Free-energy
estimates, cooperativity scores and AI predictions are computational proxies, not
experimental measurements. See the Limitations section above and
<code>docs/reproducibility.md</code> for full caveats.
</div>
</body>
</html>
"""
)


def render_html_report(data: ReportData, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _TEMPLATE.render(data=data)
    output_path.write_text(html, encoding="utf-8")
    return output_path
