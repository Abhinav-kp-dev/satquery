"""Stage 7 deliverable: an auditable PDF report of one query.

Contains the question and answer, confidence breakdown, the evidence
overlay, adjudicated claims with the ledger entries they cite, grounded
regions with coordinates, the tool plan, and the ledger entries with their
hash-chain values -- enough for a reader to re-verify every figure.
"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

from app.config import RENDER_DIR
from app.models.schemas import QueryTrace


def _t(s: object) -> str:
    """Core PDF fonts are latin-1; replace what they cannot encode."""
    text = str(s).replace("—", "-").replace("–", "-").replace("→", "->").replace("²", "2").replace("°", " deg")
    return text.encode("latin-1", "replace").decode("latin-1")


class _Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "SatQuery AI - evidence report", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, _t(f"page {self.page_no()}  |  every figure in this report is traceable to a hash-chained ledger entry"), align="C")
        self.set_text_color(0, 0, 0)


def _h(pdf: FPDF, text: str):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _t(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)


def _kv(pdf: FPDF, k: str, v: object):
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.cell(42, 5, _t(k))
    pdf.set_font("Helvetica", "", 8.5)
    pdf.multi_cell(0, 5, _t(v), new_x="LMARGIN", new_y="NEXT")


def _table(pdf: FPDF, headers: list[str], rows: list[list[object]], widths: list[float]):
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_fill_color(235, 238, 245)
    for h, w in zip(headers, widths):
        pdf.cell(w, 5, _t(h), border=1, fill=True)
    pdf.ln()
    pdf.set_font("Courier", "", 7)
    for row in rows:
        for cell, w in zip(row, widths):
            text = _t(cell)
            max_chars = max(4, int(w / 1.55))
            pdf.cell(w, 4.5, text if len(text) <= max_chars else text[: max_chars - 1] + "~", border=1)
        pdf.ln()
    pdf.set_font("Helvetica", "", 8.5)


def build_pdf(trace: QueryTrace, out_path: Path) -> Path:
    pdf = _Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, _t(trace.question), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(1)
    pdf.multi_cell(0, 5.5, _t(trace.answer), new_x="LMARGIN", new_y="NEXT")

    _h(pdf, "Summary")
    _kv(pdf, "Query id", f"{trace.query_id}  (session {trace.session_id})")
    _kv(pdf, "Created", trace.created_at)
    _kv(pdf, "Input", trace.config.value.replace("_", " "))
    if trace.spec:
        _kv(pdf, "Parsed as", f"task={trace.spec.task.value}, target={trace.spec.target.value}, metric={trace.spec.metric}")
    _kv(pdf, "Sufficiency", f"{trace.sufficiency.value}: {trace.sufficiency_reason}")
    _kv(pdf, "Confidence", f"{trace.confidence.band} ({trace.confidence.overall:.2f}) - {trace.confidence.basis}")
    for k, v in trace.confidence.components.items():
        _kv(pdf, f"  {k}", f"{v:.2f}  {trace.confidence.component_notes.get(k, '')}")
    _kv(pdf, "Narrator", f"{trace.narrator}" + (f" (adapter {trace.adapter})" if trace.adapter else ""))
    _kv(pdf, "Adjudication", trace.adjudication_status)
    _kv(pdf, "Ledger head", trace.ledger_head or "-")

    images = []
    if trace.overlay_url:
        images.append(("Evidence overlay (tool-derived masks)", RENDER_DIR / Path(trace.overlay_url).name))
    for i, u in enumerate(trace.render_urls):
        images.append((f"Input {i + 1}", RENDER_DIR / Path(u).name))
    images = [(c, p) for c, p in images if p.exists()]
    if images:
        _h(pdf, "Imagery")
        size = 88 if len(images) > 1 else 120
        x0 = pdf.get_x()
        y0 = pdf.get_y()
        for i, (cap, p) in enumerate(images[:2]):
            x = x0 + i * (size + 6)
            pdf.image(str(p), x=x, y=y0, w=size)
            if trace.regions and i == 0 and trace.overlay_url:
                pdf.set_draw_color(255, 230, 0)
                for r in trace.regions[:10]:
                    bx0, by0, bx1, by1 = r.bbox_norm
                    pdf.rect(x + bx0 * size, y0 + by0 * size, (bx1 - bx0) * size, (by1 - by0) * size)
                pdf.set_draw_color(200, 200, 200)
            pdf.set_xy(x, y0 + size + 1)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.cell(size, 4, _t(cap))
        pdf.set_xy(x0, y0 + size + 7)

    if trace.claims:
        _h(pdf, "Adjudicated claims (measurement wins on conflict)")
        _table(
            pdf, ["delivered", "claimed", "unit", "ledger entry", "ledger value", "status"],
            [[c.text, c.value, c.unit, f"{c.ledger_entry_id or '-'} {c.matched_ledger_key or ''}", c.ledger_value if c.ledger_value is not None else "-",
              "corrected" if c.corrected else "ok"] for c in trace.claims],
            [48, 18, 16, 56, 22, 20],
        )

    if trace.regions:
        _h(pdf, "Grounded regions")
        _table(
            pdf, ["region", "area ha", "bbox px", "centroid lat, lon", "ledger"],
            [[r.region_id, f"{r.area_ha:,.2f}", r.bbox_px, f"{r.centroid_lonlat[1]:.5f}, {r.centroid_lonlat[0]:.5f}" if r.centroid_lonlat else "-",
              r.ledger_entry_id or "-"] for r in trace.regions],
            [34, 22, 46, 54, 24],
        )

    if trace.stages:
        _h(pdf, "Pipeline stages")
        _table(pdf, ["stage", "status", "ms", "summary"],
               [[s.name, s.status, f"{s.runtime_ms:.1f}", s.summary] for s in trace.stages], [22, 16, 14, 128])

    if trace.ledger:
        _h(pdf, "Evidence ledger")
        _table(
            pdf, ["seq", "id", "stage", "kind", "key", "value", "hash"],
            [[e.seq, e.entry_id, e.stage, e.kind, e.key,
              (f"{e.value:.4f}" if isinstance(e.value, float) else str(e.value))[:40] + (f" {e.unit}" if e.unit and not isinstance(e.value, dict) else ""),
              e.hash[:12]] for e in trace.ledger],
            [10, 10, 18, 20, 44, 52, 26],
        )

    pdf.output(str(out_path))
    return out_path
