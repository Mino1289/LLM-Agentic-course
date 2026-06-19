"""Export report tool — Markdown / PDF report generation."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from html import escape
from typing import Any

from src.paths import REPORTS_DIR, ensure_dir
from src.tools.schemas import ExportReportArgs


def _markdown_line_to_pdf_flowable(line: str, styles: Any) -> Any:
    from reportlab.platypus import Paragraph, Spacer

    stripped = line.strip()
    if not stripped:
        return Spacer(1, 8)
    if stripped.startswith("### "):
        return Paragraph(escape(stripped[4:]), styles["Heading3"])
    if stripped.startswith("## "):
        return Paragraph(escape(stripped[3:]), styles["Heading2"])
    if stripped.startswith("# "):
        return Paragraph(escape(stripped[2:]), styles["Heading1"])
    if stripped.startswith("- "):
        return Paragraph(f"• {escape(stripped[2:])}", styles["BodyText"])
    text = escape(stripped)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    return Paragraph(text, styles["BodyText"])


def _write_pdf_report(path: Any, title: str, content: str) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires reportlab. Install with: pip install -r requirements.txt"
        ) from exc

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=48,
        title=title,
    )
    story = [_markdown_line_to_pdf_flowable(f"# {title}", styles), Spacer(1, 12)]
    for line in content.splitlines():
        story.append(_markdown_line_to_pdf_flowable(line, styles))
    doc.build(story)


def run_export_investment_report(args: ExportReportArgs) -> dict[str, Any]:
    ensure_dir(REPORTS_DIR)
    title = args.title
    content = args.content
    fmt = args.format
    safe_title = re.sub(r"[^\w\-]+", "_", title.strip())[:80] or "investment_report"
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    extension = "md" if fmt.lower() != "pdf" else "pdf"
    filename = f"{safe_title}_{timestamp}.{extension}"
    path = REPORTS_DIR / filename

    if extension == "pdf":
        _write_pdf_report(path, title, content)
    else:
        path.write_text(f"# {title}\n\n{content}", encoding="utf-8")

    return {
        "text": f"Report saved to {path}",
        "path": str(path),
        "filename": path.name,
        "format": extension,
        "title": title,
    }
