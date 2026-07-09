from __future__ import annotations

"""
Render the aggregated experiment manifests into a single self-contained
`catalog.html` dashboard — a browsable per-experiment view of how far each
recording has progressed through the pipeline:

    Recording -> SLEAP inference -> Detectron2 -> 3D reconstruction -> 3D ID

Each row is one experiment; a five-segment progress track shows the stages it
has reached (filled) versus not (empty), with a tooltip carrying the per-stage
detail (merged views, mask views, SLEAP rounds, recon configs). The page has
no external dependencies (inline CSS + a few lines of vanilla JS for
text/stage filtering) so it opens straight from the network share.

Callable from `build_catalog` (`render_catalog_html(manifests, path)`) or
standalone against an existing `catalog.jsonl`:

    python -m scripts.common.render_catalog_html \\
        --jsonl Z:\\StephensU\\medaka\\medaka_202512\\catalog.jsonl \\
        --output Z:\\StephensU\\medaka\\medaka_202512\\catalog.html
"""

import argparse
import html
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.pipeline_status import STAGE_ORDER

# Display labels for the five-segment progress track.
STAGE_LABELS = {
    "recorded": "Rec",
    "sleap_inferred": "SLEAP",
    "detectron2": "D2",
    "reconstructed_3d": "3D",
    "identified_3d": "ID",
}


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _stage_index(name: str | None) -> int:
    if name in STAGE_ORDER:
        return STAGE_ORDER.index(name)
    return -1


def _stage_track_html(pipeline: dict) -> str:
    """A five-segment coloured track for one experiment's furthest stage."""
    furthest = pipeline.get("furthest_stage")
    reached = _stage_index(furthest)
    sleap = pipeline.get("sleap_infer") or {}
    d2 = pipeline.get("detectron2") or {}
    recon = pipeline.get("reconstruction_3d") or {}

    tips = {
        "recorded": "raw multi-camera recording present",
        "sleap_inferred": (
            f"{sleap.get('n_merged_views', 0)}/3 merged views"
            + (f"; rounds: {', '.join(sleap.get('rounds') or [])}"
               if sleap.get("rounds") else "")
        ),
        "detectron2": (
            f"{d2.get('n_views', 0)}/3 views with masks"
            if d2.get("present") else "no segmentation masks"
        ),
        "reconstructed_3d": (
            f"configs: {', '.join(recon.get('configs') or []) or '—'}"
            if recon.get("has_3d_csv") or recon.get("configs")
            else "no 3D tracks"
        ),
        "identified_3d": (
            "identified 3D tracks present"
            if recon.get("has_identified_csv") else "no per-fish 3D IDs"
        ),
    }

    cells = []
    for idx, stage in enumerate(STAGE_ORDER):
        state = "on" if idx <= reached else "off"
        tip = _esc(tips.get(stage, ""))
        cells.append(
            f'<span class="seg {state}" title="{_esc(STAGE_LABELS[stage])}: {tip}">'
            f'{_esc(STAGE_LABELS[stage])}</span>'
        )
    return f'<span class="track">{"".join(cells)}</span>'


def _fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _row_html(m: dict) -> str:
    exp = m.get("experiment") or {}
    pipeline = m.get("pipeline") or {}
    sleap = pipeline.get("sleap_infer") or {}
    d2 = pipeline.get("detectron2") or {}
    recon = pipeline.get("reconstruction_3d") or {}
    boris = m.get("boris") or {}
    expl = m.get("experiment_list") or {}

    folder = m.get("folder_name", "")
    furthest = pipeline.get("furthest_stage") or "recorded"

    views = m.get("views") or {}
    top_frames = ((views.get("top_cam") or {}).get("total_frames"))

    sleap_cell = (
        f"{sleap.get('n_merged_views', 0)}/3"
        + (f'<span class="rounds">{_esc("; ".join(sleap.get("rounds") or []))}</span>'
           if sleap.get("rounds") else "")
    )
    d2_cell = f"{d2.get('n_views', 0)}/3" if d2.get("present") else "—"
    recon_cell = "—"
    if recon.get("has_3d_csv") or recon.get("configs"):
        recon_cell = _esc("; ".join(recon.get("configs") or []) or "csv")
        if recon.get("has_identified_csv"):
            recon_cell += ' <span class="idbadge">ID</span>'

    boris_cell = _fmt_int(boris.get("n_events")) if boris else "—"
    eggs = expl.get("eggs")
    accept = expl.get("female_accept")
    outcome = "—"
    if expl:
        parts = []
        if eggs not in (None, ""):
            parts.append(f"eggs {_esc(eggs)}")
        if accept not in (None, ""):
            parts.append(f"acc {_esc(accept)}")
        outcome = " ".join(parts) if parts else "—"

    return (
        f'<tr data-stage="{_esc(furthest)}" data-search="{_esc(folder.lower())}">'
        f"<td class=\"mono\">{_esc(folder)}</td>"
        f"<td>{_esc(exp.get('start_date'))}</td>"
        f"<td class=\"ctr\">{_esc(exp.get('group'))}</td>"
        f"<td class=\"ctr\">{_esc(exp.get('trial'))}/{_esc(exp.get('day'))}</td>"
        f"<td>{_stage_track_html(pipeline)}</td>"
        f"<td class=\"ctr\">{'✓' if views else '—'}</td>"
        f"<td class=\"num\">{_fmt_int(top_frames)}</td>"
        f"<td class=\"ctr\">{sleap_cell}</td>"
        f"<td class=\"ctr\">{d2_cell}</td>"
        f"<td class=\"ctr\">{recon_cell}</td>"
        f"<td class=\"num\">{boris_cell}</td>"
        f"<td class=\"ctr\">{outcome}</td>"
        f"</tr>"
    )


def _summary_counts(manifests: list[dict]) -> dict[str, int]:
    counts = {stage: 0 for stage in STAGE_ORDER}
    for m in manifests:
        furthest = (m.get("pipeline") or {}).get("furthest_stage")
        reached = _stage_index(furthest)
        for idx, stage in enumerate(STAGE_ORDER):
            if idx <= reached:
                counts[stage] += 1
    return counts


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       padding: 24px; background: #f6f7f9; color: #1b1f24; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: #6b7280; font-size: 13px; margin-bottom: 16px; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 10px 14px; min-width: 120px; }
.card .n { font-size: 22px; font-weight: 650; }
.card .l { font-size: 12px; color: #6b7280; text-transform: uppercase;
           letter-spacing: .04em; }
.toolbar { display: flex; gap: 10px; align-items: center; margin-bottom: 12px;
           flex-wrap: wrap; }
input[type=search], select { padding: 6px 10px; border: 1px solid #cbd5e1;
           border-radius: 6px; font-size: 13px; background: #fff; color: inherit; }
.table-wrap { overflow-x: auto; background: #fff; border: 1px solid #e5e7eb;
              border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid #eef0f3;
         white-space: nowrap; }
th { position: sticky; top: 0; background: #f1f3f6; font-weight: 600;
     cursor: pointer; user-select: none; }
tr:hover td { background: #fafbfc; }
.mono { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
.ctr { text-align: center; } .num { text-align: right;
       font-variant-numeric: tabular-nums; }
.track { display: inline-flex; gap: 2px; }
.seg { font-size: 10px; font-weight: 600; padding: 2px 5px; border-radius: 3px;
       min-width: 34px; text-align: center; }
.seg.on { background: #16a34a; color: #fff; }
.seg.off { background: #e5e7eb; color: #9aa3af; }
.rounds { display: block; font-size: 10px; color: #6b7280; }
.idbadge { background: #2563eb; color: #fff; border-radius: 3px; padding: 0 4px;
           font-size: 10px; }
.foot { color: #9aa3af; font-size: 12px; margin-top: 14px; }
@media (prefers-color-scheme: dark) {
  body { background: #0f1115; color: #e5e7eb; }
  .card, .table-wrap, th, input, select { background: #1a1d23;
       border-color: #2a2f37; }
  th { background: #20242b; } tr:hover td { background: #1e2229; }
  td { border-color: #23272e; }
  .seg.off { background: #2a2f37; color: #6b7280; }
}
"""

_JS = """
const q = document.getElementById('q');
const stageSel = document.getElementById('stage');
const rows = Array.from(document.querySelectorAll('tbody tr'));
function apply() {
  const term = q.value.trim().toLowerCase();
  const stage = stageSel.value;
  const want = STAGE_ORDER.indexOf(stage);
  let shown = 0;
  for (const r of rows) {
    const okText = !term || r.dataset.search.includes(term);
    // "reached >= selected stage" so counts match the cumulative dropdown labels
    const okStage = stage === 'all' ||
      STAGE_ORDER.indexOf(r.dataset.stage) >= want;
    const show = okText && okStage;
    r.style.display = show ? '' : 'none';
    if (show) shown++;
  }
  document.getElementById('count').textContent = shown + ' / ' + rows.length;
}
q.addEventListener('input', apply);
stageSel.addEventListener('change', apply);
// click-to-sort on header cells
document.querySelectorAll('th').forEach((th, i) => th.addEventListener('click', () => {
  const tbody = document.querySelector('tbody');
  const asc = th.dataset.asc !== 'true'; th.dataset.asc = asc;
  const sorted = rows.slice().sort((a, b) => {
    const av = a.children[i].textContent.trim();
    const bv = b.children[i].textContent.trim();
    const an = parseFloat(av.replace(/[^0-9.-]/g, ''));
    const bn = parseFloat(bv.replace(/[^0-9.-]/g, ''));
    const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : av.localeCompare(bv);
    return asc ? cmp : -cmp;
  });
  sorted.forEach(r => tbody.appendChild(r));
}));
apply();
"""


def render_catalog_html(manifests: list[dict], output_path: str | Path) -> Path:
    """Write a self-contained catalog.html dashboard for the given manifests."""
    output_path = Path(output_path)
    ordered = sorted(manifests, key=lambda m: m.get("folder_name", ""))
    counts = _summary_counts(ordered)

    cards = "".join(
        f'<div class="card"><div class="n">{counts[stage]}</div>'
        f'<div class="l">{_esc(STAGE_LABELS[stage])}</div></div>'
        for stage in STAGE_ORDER
    )
    # counts are cumulative ("reached at least this stage"), so label with >=
    stage_options = "".join(
        f'<option value="{_esc(stage)}">&ge; {_esc(STAGE_LABELS[stage])} '
        f'({counts[stage]})</option>'
        for stage in STAGE_ORDER
    )
    stage_order_js = json.dumps(list(STAGE_ORDER))
    rows_html = "\n".join(_row_html(m) for m in ordered)

    headers = [
        "Experiment", "Date", "Grp", "T/D", "Pipeline", "Rec",
        "Top frames", "SLEAP", "D2", "3D tracks", "BORIS", "Outcome",
    ]
    header_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>medaka_202512 experiment catalogue</title>
<style>{_CSS}</style></head><body>
<h1>medaka_202512 &mdash; experiment catalogue</h1>
<div class="sub">{len(ordered)} experiments &middot; pipeline progress:
 Recording &rarr; SLEAP &rarr; Detectron2 &rarr; 3D reconstruction &rarr; 3D ID</div>
<div class="cards">{cards}</div>
<div class="toolbar">
  <input type="search" id="q" placeholder="filter by experiment name&hellip;">
  <select id="stage"><option value="all">all stages</option>{stage_options}</select>
  <span class="sub" id="count"></span>
</div>
<div class="table-wrap"><table>
<thead><tr>{header_html}</tr></thead>
<tbody>
{rows_html}
</tbody></table></div>
<div class="foot">Generated by scripts.common.render_catalog_html.
 Filled segments = stage reached. Hover a segment for detail.</div>
<script>const STAGE_ORDER = {stage_order_js};
{_JS}</script>
</body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path


def _load_jsonl(path: Path) -> list[dict]:
    manifests: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                manifests.append(json.loads(line))
    return manifests


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render catalog.jsonl into a browsable catalog.html dashboard."
    )
    parser.add_argument(
        "--jsonl", required=True, help="Path to catalog.jsonl produced by build_catalog."
    )
    parser.add_argument(
        "--output", default=None, help="Output HTML path (default: alongside jsonl)."
    )
    args = parser.parse_args(argv)

    jsonl_path = Path(args.jsonl)
    manifests = _load_jsonl(jsonl_path)
    out = Path(args.output) if args.output else jsonl_path.with_suffix(".html")
    render_catalog_html(manifests, out)
    print(f"Wrote {out} ({len(manifests)} experiments)")


if __name__ == "__main__":
    main()
