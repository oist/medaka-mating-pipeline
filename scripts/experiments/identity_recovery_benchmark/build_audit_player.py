from __future__ import annotations

"""Build a local, method-blinded video-rating page for the crossing audit."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "event_id",
    "blinded_method",
    "clip",
    "continuity",
    "localization",
    "confidence_1_to_3",
    "notes",
]


def _presentation_order(table: pd.DataFrame, seed: int) -> list[int]:
    """Randomize clips while avoiding adjacent versions of the same event."""
    rng = np.random.default_rng(seed)
    remaining = list(range(len(table)))
    order: list[int] = []
    previous_event: str | None = None
    while remaining:
        candidates = [
            index
            for index in remaining
            if str(table.iloc[index]["event_id"]) != previous_event
        ]
        if not candidates:
            candidates = remaining
        chosen = int(rng.choice(candidates))
        order.append(chosen)
        previous_event = str(table.iloc[chosen]["event_id"])
        remaining.remove(chosen)
    return order


def write_audit_player(
    worksheet: Path,
    output: Path,
    *,
    seed: int = 20260714,
) -> Path:
    table = pd.read_csv(worksheet, keep_default_na=False)
    missing = set(REQUIRED_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"worksheet missing {sorted(missing)}")
    if table.duplicated(["event_id", "blinded_method", "clip"]).any():
        raise ValueError("worksheet audit rows are not unique")

    items = []
    for audit_index, original_index in enumerate(_presentation_order(table, seed), start=1):
        row = table.iloc[original_index]
        items.append(
            {
                "audit_id": f"R{audit_index:03d}",
                "original_index": int(original_index),
                **{column: str(row[column]) for column in REQUIRED_COLUMNS},
            }
        )
    payload = json.dumps(items, ensure_ascii=True).replace("<", "\\u003c")
    html = HTML_TEMPLATE.replace("__AUDIT_DATA__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(write_audit_player(args.worksheet, args.output, seed=args.seed))


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blinded medaka crossing audit</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; background:#10151b; color:#edf2f7; }
    * { box-sizing:border-box; }
    body { margin:0; }
    header { position:sticky; top:0; z-index:2; display:flex; gap:18px; align-items:center; justify-content:space-between; padding:14px 22px; background:#17202a; border-bottom:1px solid #34495e; }
    header h1 { margin:0; font-size:20px; }
    #progress { color:#a9cce3; font-variant-numeric:tabular-nums; }
    main { width:min(1380px, 100%); margin:auto; padding:22px; display:grid; grid-template-columns:minmax(0, 1.55fr) minmax(330px, .85fr); gap:24px; }
    video { width:100%; max-height:78vh; border-radius:10px; background:#000; box-shadow:0 12px 40px #0008; }
    #video-status { min-height:24px; padding:8px 2px; color:#f8c471; font-size:13px; }
    .panel { background:#17202a; border:1px solid #34495e; border-radius:12px; padding:18px; }
    .panel h2 { margin:0 0 4px; font-size:20px; }
    .subtle { color:#aab7b8; margin:0 0 18px; }
    fieldset { border:0; border-top:1px solid #34495e; padding:16px 0 0; margin:16px 0 0; }
    legend { padding:0 8px 0 0; font-weight:700; }
    .definition { color:#bdc3c7; font-size:13px; margin:5px 0 10px; line-height:1.35; }
    .options { display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; }
    button { appearance:none; border:1px solid #5d6d7e; border-radius:8px; padding:10px 8px; background:#212f3d; color:#edf2f7; font-weight:650; cursor:pointer; }
    button:hover { border-color:#85c1e9; }
    button.selected { background:#117864; border-color:#58d68d; }
    .navigation { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:18px; }
    textarea { width:100%; min-height:74px; resize:vertical; border-radius:8px; border:1px solid #5d6d7e; background:#101820; color:#fff; padding:10px; }
    .export { width:100%; margin-top:12px; background:#1f618d; border-color:#5dade2; }
    #message { min-height:20px; color:#82e0aa; font-size:13px; margin-top:8px; }
    .warning { margin:0 0 18px; padding:10px 12px; border-left:4px solid #f5b041; background:#2b251b; color:#f8c471; }
    @media (max-width:900px) { main { grid-template-columns:1fr; } video { max-height:62vh; } header { align-items:flex-start; flex-direction:column; gap:4px; } }
  </style>
</head>
<body>
  <header>
    <h1>Blinded medaka crossing audit</h1>
    <div id="progress">0 rated</div>
  </header>
  <main>
    <section><video id="video" controls playsinline preload="metadata"></video><div id="video-status" role="status"></div></section>
    <section class="panel">
      <p class="warning">Keep <code>PRIVATE_BLINDING_KEY.csv</code> closed until the CSV is exported.</p>
      <h2 id="audit-title">Rating</h2>
      <p class="subtle" id="clip-meta"></p>
      <fieldset>
        <legend>Identity continuity</legend>
        <p class="definition">Pass: every visible T-label stays on the same biological fish. Switch: any label changes fish. Use uncertain only when the video cannot decide.</p>
        <div class="options">
          <button data-field="continuity" data-value="pass">Pass</button>
          <button data-field="continuity" data-value="switch">Switch</button>
          <button data-field="continuity" data-value="uncertain">Uncertain</button>
        </div>
      </fieldset>
      <fieldset>
        <legend>Fish localization</legend>
        <p class="definition">Pass: all three labels are on real fish before and after the crossing. Fail: a label is missing or visibly off-fish.</p>
        <div class="options">
          <button data-field="localization" data-value="pass">Pass</button>
          <button data-field="localization" data-value="fail">Fail</button>
          <button data-field="localization" data-value="uncertain">Uncertain</button>
        </div>
      </fieldset>
      <fieldset>
        <legend>Confidence</legend>
        <p class="definition">1 = low, 2 = medium, 3 = high.</p>
        <div class="options">
          <button data-field="confidence_1_to_3" data-value="1">1</button>
          <button data-field="confidence_1_to_3" data-value="2">2</button>
          <button data-field="confidence_1_to_3" data-value="3">3</button>
        </div>
      </fieldset>
      <fieldset><legend>Optional notes</legend><textarea id="notes" placeholder="What made this rating clear or ambiguous?"></textarea></fieldset>
      <div class="navigation"><button id="previous">Previous</button><button id="next">Next</button></div>
      <button class="export" id="export">Download current ratings as CSV</button>
      <div id="message"></div>
    </section>
  </main>
  <script>
    "use strict";
    const items = __AUDIT_DATA__;
    const storageKey = "medaka_blinded_crossing_audit_v1";
    let cursor = 0;
    let state = {};
    try { state = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) { state = {}; }
    const keyFor = item => [item.event_id, item.blinded_method, item.clip].join("|");
    const videoElement = document.getElementById("video");
    const videoStatus = document.getElementById("video-status");
    videoElement.addEventListener("loadeddata", () => { videoStatus.textContent = ""; });
    videoElement.addEventListener("error", () => {
      const code = videoElement.error ? videoElement.error.code : "unknown";
      videoStatus.textContent = `This browser could not load the H.264 clip (media error ${code}). Confirm the clips/ folder remains beside AUDIT_PLAYER.html.`;
    });
    for (const item of items) {
      const key = keyFor(item);
      state[key] = Object.assign({
        continuity:item.continuity || "", localization:item.localization || "",
        confidence_1_to_3:item.confidence_1_to_3 || "", notes:item.notes || ""
      }, state[key] || {});
    }
    const complete = value => ["pass", "switch", "uncertain"].includes(value.continuity)
      && ["pass", "fail", "uncertain"].includes(value.localization)
      && ["1", "2", "3"].includes(String(value.confidence_1_to_3));
    function save() { try { localStorage.setItem(storageKey, JSON.stringify(state)); } catch (_) {} }
    function render() {
      const item = items[cursor], rating = state[keyFor(item)];
      document.getElementById("audit-title").textContent = `${item.audit_id} — clip ${cursor + 1} of ${items.length}`;
      document.getElementById("clip-meta").textContent = `${item.event_id}, blinded method ${item.blinded_method}`;
      const video = videoElement;
      if (video.getAttribute("src") !== item.clip) { videoStatus.textContent = "Loading clip…"; video.setAttribute("src", item.clip); video.load(); }
      document.querySelectorAll("button[data-field]").forEach(button => {
        button.classList.toggle("selected", String(rating[button.dataset.field]) === button.dataset.value);
      });
      document.getElementById("notes").value = rating.notes || "";
      document.getElementById("previous").disabled = cursor === 0;
      document.getElementById("next").disabled = cursor === items.length - 1;
      const done = items.filter(item => complete(state[keyFor(item)])).length;
      document.getElementById("progress").textContent = `${done} / ${items.length} rated`;
      document.getElementById("message").textContent = done === items.length ? "All clips rated — download the CSV and return it to Codex." : "Ratings are saved locally in this browser.";
    }
    document.querySelectorAll("button[data-field]").forEach(button => button.addEventListener("click", () => {
      const item = items[cursor]; state[keyFor(item)][button.dataset.field] = button.dataset.value; save(); render();
    }));
    document.getElementById("notes").addEventListener("input", event => { state[keyFor(items[cursor])].notes = event.target.value; save(); });
    document.getElementById("previous").addEventListener("click", () => { if (cursor > 0) { cursor--; render(); } });
    document.getElementById("next").addEventListener("click", () => { if (cursor + 1 < items.length) { cursor++; render(); } });
    const csvEscape = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
    document.getElementById("export").addEventListener("click", () => {
      const header = ["event_id","blinded_method","clip","continuity","localization","confidence_1_to_3","notes"];
      const rows = [...items].sort((a,b) => a.original_index - b.original_index).map(item => {
        const rating = state[keyFor(item)];
        return [item.event_id,item.blinded_method,item.clip,rating.continuity,rating.localization,rating.confidence_1_to_3,rating.notes].map(csvEscape).join(",");
      });
      const blob = new Blob([[header.join(","), ...rows].join("\r\n") + "\r\n"], {type:"text/csv;charset=utf-8"});
      const url = URL.createObjectURL(blob), link = document.createElement("a");
      link.href = url; link.download = "AUDIT_WORKSHEET_COMPLETED.csv"; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
