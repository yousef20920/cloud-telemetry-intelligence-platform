"""Rendering helpers for model evaluation reports."""

from __future__ import annotations

from html import escape


def render_svg_bar_chart(title: str, series: list[tuple[str, float]]) -> str:
    safe_title = escape(title)
    if not series:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="120">'
            '<text x="20" y="28" font-size="18" font-family="Arial">No feature importances available</text>'
            "</svg>"
        )

    width = 720
    row_height = 34
    height = 70 + (row_height * len(series))
    max_value = max(abs(value) for _, value in series) or 1.0
    bars: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        f'<text x="20" y="30" fill="#f8fafc" font-size="20" font-family="Arial">{safe_title}</text>',
    ]

    for index, (label, value) in enumerate(series):
        top = 60 + (index * row_height)
        bar_width = max(4.0, (abs(value) / max_value) * 360.0)
        bars.append(
            f'<text x="20" y="{top + 16}" fill="#cbd5e1" font-size="13" font-family="Arial">{escape(label)}</text>'
        )
        bars.append(
            f'<rect x="290" y="{top}" width="{bar_width:.2f}" height="18" rx="4" fill="#38bdf8"/>'
        )
        bars.append(
            f'<text x="{300 + bar_width:.2f}" y="{top + 14}" fill="#e2e8f0" font-size="12" font-family="Arial">{value:.4f}</text>'
        )

    bars.append("</svg>")
    return "".join(bars)


def render_markdown_dashboard(
    *,
    classification_rows: list[dict[str, object]],
    regression_rows: list[dict[str, object]],
    unsupervised_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# Model Evaluation Dashboard",
        "",
        "This dashboard summarizes predictive quality and operational metrics across the trained telemetry models.",
        "",
    ]

    def add_table(title: str, rows: list[dict[str, object]], columns: list[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("No models available.")
            lines.append("")
            return
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
        lines.append("")

    add_table(
        "Classification",
        classification_rows,
        ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "batch_latency_ms_mean", "throughput_rows_per_sec"],
    )
    add_table(
        "Regression",
        regression_rows,
        ["model_name", "target", "rmse", "mae", "mape", "batch_latency_ms_mean", "throughput_rows_per_sec"],
    )
    add_table(
        "Unsupervised",
        unsupervised_rows,
        ["model_name", "accuracy", "precision", "recall", "f1", "batch_latency_ms_mean", "throughput_rows_per_sec"],
    )
    return "\n".join(lines)


def render_html_dashboard(
    *,
    title: str,
    summary: dict[str, object],
    markdown_dashboard: str,
    feature_svg_paths: list[str],
) -> str:
    feature_gallery = "".join(
        f'<div class="card"><h3>{escape(path.split("/")[-1])}</h3><img src="{escape(path.split("/")[-1])}" alt="{escape(path)}"/></div>'
        for path in feature_svg_paths
    )
    preformatted = escape(markdown_dashboard)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #020617;
      --panel: #0f172a;
      --panel-alt: #111827;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --border: #1e293b;
    }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: radial-gradient(circle at top left, #0f172a, #020617 55%);
      color: var(--text);
      line-height: 1.5;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 40px 24px 64px;
    }}
    .hero {{
      padding: 28px;
      border: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.92);
      border-radius: 20px;
      margin-bottom: 24px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 12px;
      display: block;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric strong {{
      color: var(--accent);
      font-size: 26px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      overflow: hidden;
    }}
    img {{
      max-width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #020617;
    }}
    pre {{
      white-space: pre-wrap;
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{escape(title)}</h1>
      <p>Predictive quality and operational analysis for the current telemetry models.</p>
      <div class="metrics">
        <div class="metric"><span>Rows</span><strong>{summary.get("row_count", 0)}</strong></div>
        <div class="metric"><span>Features</span><strong>{summary.get("feature_count", 0)}</strong></div>
        <div class="metric"><span>Classifiers</span><strong>{summary.get("classification_count", 0)}</strong></div>
        <div class="metric"><span>Regressors</span><strong>{summary.get("regression_count", 0)}</strong></div>
        <div class="metric"><span>Unsupervised</span><strong>{summary.get("unsupervised_count", 0)}</strong></div>
      </div>
    </section>
    <section class="grid">
      <div class="card">
        <h2>Comparison Tables</h2>
        <pre>{preformatted}</pre>
      </div>
      <div class="card">
        <h2>Feature Importance Plots</h2>
        <div class="grid">
          {feature_gallery or '<p>No feature-importance visualizations were available.</p>'}
        </div>
      </div>
    </section>
  </main>
</body>
</html>"""

