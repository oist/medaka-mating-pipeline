import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const HERE = path.dirname(__filename);
const DATA_DIR = path.join(HERE, "data");
const PREVIEW_DIR = path.join(HERE, "preview");
const LAYOUT_DIR = path.join(HERE, "layout");
const FINAL_PPTX = path.join(HERE, "medaka_boris_cohort_figures_20260630.pptx");

const W = 1280;
const H = 720;
const M = 52;
const INK = "#111827";
const MUTED = "#5B6472";
const PANEL = "#F2F4F7";
const RULE = "#C8CDD5";
const BLUE = "#1677B9";
const RED = "#B94040";
const GREEN = "#218B45";
const AMBER = "#B56A00";
const PURPLE = "#6F3FA0";

const CATEGORY_COLORS = {
  Courtship: BLUE,
  Competitive: RED,
  Mating: GREEN,
  Other: "#8A8F98",
};

async function readJson(file) {
  return JSON.parse(await fs.readFile(path.join(DATA_DIR, file), "utf8"));
}

function shortExp(name) {
  return String(name).replace(/^2025|^2026/, "").replace("_FCMC", "");
}

function minutes(s) {
  const v = Number(s);
  return Number.isFinite(v) ? v / 60 : 0;
}

function fmt(value, digits = 1) {
  const v = Number(value);
  return Number.isFinite(v) ? v.toFixed(digits) : "NA";
}

function pct(value, digits = 0) {
  const v = Number(value);
  return Number.isFinite(v) ? `${(100 * v).toFixed(digits)}%` : "NA";
}

function addText(slide, text, box, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: box,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = String(text ?? "");
  shape.text.style = {
    fontSize: style.fontSize ?? 20,
    bold: style.bold ?? false,
    color: style.color ?? INK,
    alignment: style.alignment ?? "left",
  };
  return shape;
}

function addPanel(slide, box, fill = PANEL) {
  return slide.shapes.add({
    geometry: "rect",
    position: box,
    fill,
    line: { style: "solid", fill: RULE, width: 1 },
  });
}

function addRule(slide, x, y, w, color = RULE) {
  slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height: 1.5 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function addHeader(slide, section, title, slideNo) {
  addText(slide, section, { left: M, top: 28, width: 420, height: 28 }, {
    fontSize: 15,
    bold: true,
    color: MUTED,
  });
  addText(slide, title, { left: M, top: 64, width: 1040, height: 74 }, {
    fontSize: 36,
    bold: true,
  });
  addText(slide, String(slideNo), { left: 1208, top: 34, width: 28, height: 26 }, {
    fontSize: 15,
    color: MUTED,
    alignment: "right",
  });
  addRule(slide, M, 146, W - 2 * M, RULE);
}

function addFooter(slide, text) {
  addText(slide, text, { left: M, top: 672, width: W - 2 * M, height: 24 }, {
    fontSize: 12,
    color: MUTED,
  });
}

function metric(slide, label, value, box, color = INK) {
  addText(slide, String(value), { left: box.left, top: box.top, width: box.width, height: 55 }, {
    fontSize: 42,
    bold: true,
    color,
  });
  addText(slide, label, { left: box.left, top: box.top + 62, width: box.width, height: 62 }, {
    fontSize: 16,
    color: MUTED,
  });
}

function addCaption(slide, headline, detail) {
  addPanel(slide, { left: 780, top: 178, width: 440, height: 416 }, "#F8F9FB");
  addText(slide, headline, { left: 812, top: 212, width: 372, height: 92 }, {
    fontSize: 25,
    bold: true,
  });
  addText(slide, detail, { left: 812, top: 324, width: 372, height: 230 }, {
    fontSize: 18,
    color: MUTED,
  });
}

function chartAxis(title) {
  return {
    title: { text: title, textStyle: { fontSize: 12, fill: MUTED } },
    textStyle: { fontSize: 10, fill: MUTED },
    majorGridlines: { style: "solid", fill: "#E6E9EF", width: 1 },
    line: { style: "solid", fill: RULE, width: 1 },
  };
}

function buildCover(p, d) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  addText(slide, "Medaka BORIS cohort", { left: M, top: 90, width: 980, height: 72 }, {
    fontSize: 52,
    bold: true,
  });
  addText(slide, "Manual ethogram figure supplement for poster selection", { left: M, top: 178, width: 900, height: 46 }, {
    fontSize: 25,
    color: MUTED,
  });
  addRule(slide, M, 260, W - 2 * M, BLUE);

  const m = d.metrics;
  metric(slide, "user-provided BORIS CSV files parsed", m.parsed_file_count, { left: M, top: 326, width: 230, height: 140 }, BLUE);
  metric(slide, "state + point events after pairing START/STOP rows", m.parsed_events, { left: 330, top: 326, width: 270, height: 140 }, INK);
  metric(slide, "experiments containing a Spawn point event", m.experiments_with_spawn, { left: 658, top: 326, width: 250, height: 140 }, GREEN);
  metric(slide, "minutes of annotated observation", fmt(m.total_observation_min, 0), { left: 960, top: 326, width: 240, height: 140 }, PURPLE);

  addText(slide,
    `Date range: ${m.date_range?.[0] ?? "NA"} to ${m.date_range?.[1] ?? "NA"}  |  ` +
    `Raw BORIS rows: ${m.raw_rows}  |  Output tables: data/*.csv + data/*.json`,
    { left: M, top: 596, width: 1120, height: 38 },
    { fontSize: 16, color: MUTED },
  );
  addFooter(slide, "Source: Drive-backed BORIS CSVs supplied in the request. Parser: build_boris_cohort_data.py.");
}

function buildScopeSlide(p, d, n) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  addHeader(slide, "B1 · Cohort Scope", "What the manual ethogram cohort contains", n);
  const m = d.metrics;
  const cards = [
    ["Courtship", `${fmt(m.total_courtship_min, 1)} min`, CATEGORY_COLORS.Courtship],
    ["Competitive", `${fmt(m.total_competitive_min, 1)} min`, CATEGORY_COLORS.Competitive],
    ["Mating", `${fmt(m.total_mating_min, 1)} min`, CATEGORY_COLORS.Mating],
    ["Wraps accepted/rejected", `${m.wrap_accepted_n} / ${m.wrap_rejected_n}`, AMBER],
  ];
  cards.forEach((c, i) => {
    const x = M + i * 292;
    addPanel(slide, { left: x, top: 174, width: 250, height: 170 }, "#F8F9FB");
    metric(slide, c[0], c[1], { left: x + 24, top: 210, width: 205, height: 116 }, c[2]);
  });

  const top = (m.top_behaviors_by_duration ?? []).slice(0, 5)
    .map((r, i) => `${i + 1}. ${r.behavior} (${r.category}) — ${fmt(r.total_s / 60, 1)} min, ${r.events} events`)
    .join("\n");
  addPanel(slide, { left: M, top: 392, width: 552, height: 224 }, PANEL);
  addText(slide, "Top behaviors by annotated duration", { left: M + 28, top: 420, width: 500, height: 32 }, {
    fontSize: 22,
    bold: true,
  });
  addText(slide, top, { left: M + 28, top: 466, width: 500, height: 132 }, {
    fontSize: 17,
    color: MUTED,
  });

  addPanel(slide, { left: 680, top: 392, width: 540, height: 224 }, "#F8F9FB");
  addText(slide, "How to use this supplement", { left: 708, top: 420, width: 480, height: 32 }, {
    fontSize: 22,
    bold: true,
  });
  addText(slide,
    "These slides broaden the poster beyond the N=3 tracking-heavy examples. Use them to pick a " +
    "cohort-level ethogram plot, then pair it with the individual-ID tracking panels from the main poster kit.",
    { left: 708, top: 466, width: 470, height: 120 },
    { fontSize: 18, color: MUTED },
  );
  addFooter(slide, "Data files: boris_cohort_events.csv, boris_cohort_experiment_summary.csv, boris_cohort_behavior_summary.csv.");
}

function buildCategoryByExperiment(p, d, n) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  addHeader(slide, "B2 · Experiment-Level Ethograms", "Annotated behavior time by experiment", n);
  const exps = d.experimentSummary;
  const categories = exps.map((r) => shortExp(r.experiment));
  slide.charts.add("bar", {
    position: { left: 56, top: 166, width: 680, height: 440 },
    categories,
    series: [
      { name: "Courtship", values: exps.map((r) => minutes(r.courtship_duration_s)), fill: CATEGORY_COLORS.Courtship },
      { name: "Competitive", values: exps.map((r) => minutes(r.competitive_duration_s)), fill: CATEGORY_COLORS.Competitive },
      { name: "Mating", values: exps.map((r) => minutes(r.mating_duration_s)), fill: CATEGORY_COLORS.Mating },
      { name: "Other", values: exps.map((r) => minutes(r.other_duration_s)), fill: CATEGORY_COLORS.Other },
    ],
    barOptions: { direction: "column", grouping: "stacked", gapWidth: 35 },
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 12, fill: MUTED } },
    xAxis: { textStyle: { fontSize: 9, fill: MUTED }, line: { style: "solid", fill: RULE, width: 1 } },
    yAxis: chartAxis("annotated state duration (min)"),
  });
  addCaption(
    slide,
    "Cohort-level behavior intensity",
    "Each bar is one observation CSV. Durations are paired START/STOP states; POINT events are retained in the tables but do not add duration. This plot is useful when choosing which trials to showcase as behavior-rich examples.",
  );
  addFooter(slide, "Figure B2 source: boris_cohort_experiment_summary.csv.");
}

function buildBehaviorRepertoire(p, d, n) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  addHeader(slide, "B3 · Behavior Repertoire", "Which manual behaviors dominate the ethogram?", n);
  const top = d.behaviorSummary.slice(0, 12).reverse();
  slide.charts.add("bar", {
    position: { left: 72, top: 160, width: 690, height: 464 },
    categories: top.map((r) => r.behavior),
    series: [{ name: "Total duration", values: top.map((r) => minutes(r.total_s)), fill: INK }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 36 },
    hasLegend: false,
    xAxis: chartAxis("total duration (min)"),
    yAxis: { textStyle: { fontSize: 12, fill: MUTED }, line: { style: "solid", fill: RULE, width: 1 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 11, fill: INK, bold: true } },
  });
  const detail = top.slice().reverse().slice(0, 5)
    .map((r) => `${r.behavior}: ${r.events} events in ${r.experiments} experiments`)
    .join("\n");
  addCaption(
    slide,
    "A compact menu of candidate biology panels",
    `Top behaviors combine frequency and duration. The five largest are:\n\n${detail}`,
  );
  addFooter(slide, "Figure B3 source: boris_cohort_behavior_summary.csv.");
}

function buildMaleAsymmetry(p, d, n) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  addHeader(slide, "B4 · Manual Male-To-Female Asymmetry", "BORIS m1tf/m2tf duration by experiment", n);
  const exps = d.experimentSummary.filter((r) => Number(r.m1tf_duration_s) > 0 || Number(r.m2tf_duration_s) > 0);
  slide.charts.add("bar", {
    position: { left: 58, top: 166, width: 688, height: 430 },
    categories: exps.map((r) => shortExp(r.experiment)),
    series: [
      { name: "m1 -> female", values: exps.map((r) => minutes(r.m1tf_duration_s)), fill: BLUE },
      { name: "m2 -> female", values: exps.map((r) => minutes(r.m2tf_duration_s)), fill: "#7A7F87" },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 42 },
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 12, fill: MUTED } },
    xAxis: { textStyle: { fontSize: 9, fill: MUTED }, line: { style: "solid", fill: RULE, width: 1 } },
    yAxis: chartAxis("duration (min)"),
  });
  const strong = exps
    .filter((r) => Number.isFinite(Number(r.boris_male_courtship_ratio)))
    .sort((a, b) => Number(b.boris_male_courtship_ratio) - Number(a.boris_male_courtship_ratio))
    .slice(0, 4)
    .map((r) => `${shortExp(r.experiment)}: ${fmt(r.boris_male_courtship_ratio, 1)}x toward ${r.boris_more_courtship_subject}`)
    .join("\n");
  addCaption(
    slide,
    "Independent manual asymmetry check",
    `The manual labels are annotator-defined, so they are not automatically the same as appearance M1/M2. But asymmetric male-to-female courtship is still a useful cross-check.\n\nLargest asymmetries:\n${strong}`,
  );
  addFooter(slide, "Figure B4 source: boris_cohort_experiment_summary.csv.");
}

function buildCourtshipCompetition(p, d, n) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  addHeader(slide, "B5 · Courtship Versus Competition", "Courtship and competition occupy different lanes", n);
  const exps = d.experimentSummary;
  slide.charts.add("bar", {
    position: { left: 58, top: 166, width: 688, height: 430 },
    categories: exps.map((r) => shortExp(r.experiment)),
    series: [
      { name: "Male -> female", values: exps.map((r) => minutes(r.male_to_female_duration_s)), fill: BLUE },
      { name: "Male-male", values: exps.map((r) => minutes(r.male_male_duration_s)), fill: RED },
      { name: "Mating", values: exps.map((r) => minutes(r.mating_duration_s)), fill: GREEN },
    ],
    barOptions: { direction: "column", grouping: "stacked", gapWidth: 35 },
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 12, fill: MUTED } },
    xAxis: { textStyle: { fontSize: 9, fill: MUTED }, line: { style: "solid", fill: RULE, width: 1 } },
    yAxis: chartAxis("duration (min)"),
  });
  addCaption(
    slide,
    "A bridge from ethogram to tracking",
    "The poster can use this plot to show why individual-ID tracking matters: manual coding captures the event class, while the tracking pipeline assigns those behaviors to a specific fish trajectory.",
  );
  addFooter(slide, "Figure B5 source: boris_cohort_experiment_summary.csv.");
}

function buildMatingEndpoints(p, d, n) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  addHeader(slide, "B6 · Mating Endpoints", "Spawn and wrap annotations identify the decision endpoint", n);
  const exps = d.experimentSummary.filter((r) => r.has_spawn || Number(r.wrap_events) > 0);
  const values = [["Experiment", "Spawn (s)", "Wraps", "Accepted", "Rejected"]];
  for (const r of exps) {
    values.push([
      shortExp(r.experiment),
      Number.isFinite(Number(r.spawn_time_s)) ? fmt(r.spawn_time_s, 1) : "",
      String(r.wrap_events ?? 0),
      String(r.wrap_accepted_n ?? 0),
      String(r.wrap_rejected_n ?? 0),
    ]);
  }
  const rows = Math.min(values.length, 13);
  const table = slide.tables.add({
    rows,
    columns: 5,
    left: 62,
    top: 176,
    width: 680,
    height: 382,
    values: values.slice(0, rows),
    columnTracks: [
      { mode: "fr", value: 2.2 },
      { mode: "fr", value: 1.1 },
      { mode: "fr", value: 0.9 },
      { mode: "fr", value: 1.0 },
      { mode: "fr", value: 1.0 },
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: RULE, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: rows, columnCount: 5 }).assign({
    textStyle: { fontSize: 12, color: INK },
    margins: { left: 6, right: 6, top: 3, bottom: 3 },
  });
  for (let c = 0; c < 5; c++) {
    table.getCell(0, c).fill = INK;
    table.getCell(0, c).text.style = { fontSize: 13, bold: true, color: "#FFFFFF" };
  }
  for (let r = 0; r < rows; r++) {
    table.rows[r].height = r === 0 ? 28 : 27;
  }
  addCaption(
    slide,
    "Wrap duration separates accepted and rejected bouts",
    `Across the cohort, accepted wraps: ${d.metrics.wrap_accepted_n}; rejected wraps: ${d.metrics.wrap_rejected_n}. Median accepted duration is ${fmt(d.metrics.wrap_accepted_median_s, 1)} s versus ${fmt(d.metrics.wrap_rejected_median_s, 1)} s rejected. Full endpoint table is in the CSV.`,
  );
  addFooter(slide, "Figure B6 source: boris_cohort_experiment_summary.csv and boris_cohort_events.csv.");
}

function buildTiming(p, d, n) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  addHeader(slide, "B7 · Timing", "Ready and Spawn point events locate the decision window", n);
  const exps = d.experimentSummary.filter((r) => Number.isFinite(Number(r.ready_time_s)) || Number.isFinite(Number(r.spawn_time_s)));
  slide.charts.add("bar", {
    position: { left: 58, top: 166, width: 688, height: 430 },
    categories: exps.map((r) => shortExp(r.experiment)),
    series: [
      { name: "Ready time", values: exps.map((r) => Number.isFinite(Number(r.ready_time_s)) ? Number(r.ready_time_s) / 60 : 0), fill: AMBER },
      { name: "Spawn time", values: exps.map((r) => Number.isFinite(Number(r.spawn_time_s)) ? Number(r.spawn_time_s) / 60 : 0), fill: GREEN },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 42 },
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 12, fill: MUTED } },
    xAxis: { textStyle: { fontSize: 9, fill: MUTED }, line: { style: "solid", fill: RULE, width: 1 } },
    yAxis: chartAxis("time from recording start (min)"),
  });
  addCaption(
    slide,
    "Use timing plots to choose example windows",
    "Bars at zero indicate that a point event was not present in that observation. For tracking figures, prioritize trials with both a Spawn endpoint and rich preceding courtship/competition.",
  );
  addFooter(slide, "Figure B7 source: boris_cohort_experiment_summary.csv.");
}

function buildSelectionGuide(p, d, n) {
  const slide = p.slides.add();
  slide.background.fill = "#FFFFFF";
  addHeader(slide, "Selection Guide", "How to combine this supplement with the main poster kit", n);
  const values = [
    ["Need", "Use from this deck", "Pair with main poster kit"],
    ["Show cohort scope", "B1 scope or B2 stacked ethogram", "Section 1 setup / N=3 framing"],
    ["Show behavior repertoire", "B3 behavior duration ranking", "BORIS-vs-distance panel"],
    ["Validate chosen male manually", "B4 m1tf/m2tf asymmetry", "N=3 appearance mate-choice panels"],
    ["Separate biology mechanisms", "B5 courtship vs male-male", "Courtship vs aggression N=3 plot"],
    ["Choose endpoint examples", "B6/B7 spawn and wrap timing", "Spawn close-up / trajectory zoom"],
  ];
  const table = slide.tables.add({
    rows: values.length,
    columns: 3,
    left: 70,
    top: 176,
    width: 1110,
    height: 382,
    values,
    columnTracks: [
      { mode: "fr", value: 1.25 },
      { mode: "fr", value: 1.75 },
      { mode: "fr", value: 2.0 },
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: RULE, width: 1 });
  for (let c = 0; c < 3; c++) {
    table.getCell(0, c).fill = BLUE;
    table.getCell(0, c).text.style = { fontSize: 14, bold: true, color: "#FFFFFF" };
  }
  addFooter(slide, "Figure manifest: data/boris_cohort_figure_manifest.csv.");
}

async function writeBlob(outPath, blob) {
  await fs.writeFile(outPath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });
  const d = await readJson("boris_cohort_deck_data.json");
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  buildCover(presentation, d);
  buildScopeSlide(presentation, d, 1);
  buildCategoryByExperiment(presentation, d, 2);
  buildBehaviorRepertoire(presentation, d, 3);
  buildMaleAsymmetry(presentation, d, 4);
  buildCourtshipCompetition(presentation, d, 5);
  buildMatingEndpoints(presentation, d, 6);
  buildTiming(presentation, d, 7);
  buildSelectionGuide(presentation, d, 8);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide_${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(PREVIEW_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(LAYOUT_DIR, `${stem}.layout.json`), await layout.text());
  }
  await writeBlob(path.join(PREVIEW_DIR, "boris_cohort_montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
  const snapshot = await presentation.inspect({ kind: "slide,textbox,shape,chart,table,layout", maxChars: 12000 });
  await fs.writeFile(path.join(LAYOUT_DIR, "inspect.ndjson"), snapshot.ndjson);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(`wrote ${FINAL_PPTX}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
