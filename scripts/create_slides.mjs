// Build the six-slide, five-minute presentation (J3) from the shared facts file.
//
// Run from the project root, after the analysis reports exist:
//
//   python scripts/create_submission.py facts
//   node scripts/create_slides.mjs [--facts PATH] [--output PATH] [--replace] [--scale N]
//
// Every number comes from tmp/artifacts/submission_facts.json, which
// create_submission.py builds from reports/. The script refuses to run if any
// report changed after the facts were written.
//
// Output: output/slides/presentation.pptx. If that file exists, the validated
// deck goes to output/slides/presentation-<stamp>.pptx instead; with --replace
// the canonical file is replaced by the validated bytes. Drafts, renders,
// layout JSON and validation receipts go to tmp/artifacts/slides/run-<stamp>/.
//
// Environment overrides: RUNTIME_NODE_MODULES, RUNTIME_PYTHON, PRESENTATIONS_SKILL_DIR.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RUNTIME = "/Users/samopk/.cache/codex-runtimes/codex-primary-runtime/dependencies";
process.env.RUNTIME_NODE_MODULES ??= path.join(RUNTIME, "node/node_modules");
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON ?? path.join(RUNTIME, "python/bin/python3");
const SKILL_DIR = process.env.PRESENTATIONS_SKILL_DIR ??
  "/Users/samopk/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations";

const runtimeRequire = createRequire(path.join(process.env.RUNTIME_NODE_MODULES, "__runtime__.cjs"));
const { FileBlob, Presentation, PresentationFile } = await import(
  pathToFileURL(runtimeRequire.resolve("@oai/artifact-tool")).href);
const { applyPresentationChartFont, finalizePresentation, resolvePresentationFont } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);

const FACTS_SCHEMA = "ds5020-submission-facts.v1";
const SLIDE = { width: 1280, height: 720 };
const SLIDE_EMU = "12192000,6858000";
const MARGIN = 64;
const CONTENT_WIDTH = SLIDE.width - 2 * MARGIN;
const COLOR = {
  navy: "#1B2A4A", teal: "#0F7C80", coverTeal: "#7FD1CF", ink: "#1F2937",
  muted: "#4B5563", rule: "#D1D5DB", white: "#FFFFFF", band: "#F3F6F8",
};
const SIZE = { cover: 64, coverSub: 30, title: 44, body: 23, small: 20, source: 14 };
const NOTES_WORDS = { min: 560, max: 640, hardMin: 520, hardMax: 680 };
const FONT = resolvePresentationFont();
// Alex Rees did the original data work and exploratory SVD; Jiaxiao Chen completed the ML work.
const DATA_ENGINEER = "Alex Rees, Data Engineer";
const ML_ENGINEER = "Jiaxiao Chen, ML Engineer";

// --------------------------------------------------------------------------
// Inputs

const args = parseArgs({
  options: {
    facts: { type: "string", default: path.join(ROOT, "tmp/artifacts/submission_facts.json") },
    output: { type: "string", default: path.join(ROOT, "output/slides/presentation.pptx") },
    replace: { type: "boolean", default: false },
    "draft-only": { type: "boolean", default: false },
    scale: { type: "string", default: "1" },
    help: { type: "boolean", default: false },
  },
  strict: true,
}).values;
if (args.help) {
  console.log("node scripts/create_slides.mjs [--facts PATH] [--output PATH] [--replace] [--draft-only] [--scale N]\n" +
    "  --draft-only  validate and render privately under tmp/artifacts/slides/ without touching output/");
  process.exit(0);
}

function fail(message) {
  console.error(`error: ${message}`);
  process.exit(2);
}

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

async function loadFacts(file) {
  const raw = await fs.readFile(file, "utf8").catch(() => undefined);
  if (raw === undefined) {
    fail(`${path.relative(ROOT, file)} not found. Run \`python scripts/create_submission.py facts\` first ` +
      "(it needs reports/results.json, comparison.csv, svd_curve.csv, segments.csv and figures/customer_map.png).");
  }
  const facts = JSON.parse(raw);
  if (facts.schema !== FACTS_SCHEMA) fail(`Unexpected facts schema ${facts.schema}`);
  for (const [key, input] of Object.entries(facts.inputs ?? {})) {
    const bytes = await fs.readFile(path.join(ROOT, input.path)).catch(() => undefined);
    if (!bytes) fail(`Input ${input.path} (${key}) disappeared since the facts were written`);
    if (sha256(bytes) !== input.sha256) {
      fail(`${input.path} changed after the facts were written. Rerun \`python scripts/create_submission.py facts\`.`);
    }
  }
  return facts;
}

const facts = await loadFacts(path.resolve(args.facts));
const f = facts.fmt;
const o = facts.outcomes;
const scale = Number.parseFloat(args.scale);
if (!Number.isFinite(scale) || scale < 1) fail("--scale must be a number of at least 1");

const mapBytes = await fs.readFile(path.join(ROOT, facts.customer_map.path));
if (sha256(mapBytes) !== facts.customer_map.sha256) fail("customer_map.png changed after the facts were written");

// --------------------------------------------------------------------------
// Text helpers. Every visible string passes through these so the privacy scan
// sees exactly what the deck contains.

const deckText = [];
const notesBySlide = [];

function run(textValue, { size = SIZE.body, color = COLOR.ink, bold = false } = {}) {
  deckText.push(textValue);
  return { run: textValue, style: { typeface: FONT, fontSize: `${size}px`, color, bold } };
}

function addText(slide, name, box, paragraphs, { size = SIZE.body, color = COLOR.ink, bold = false,
  align = "left", valign = "top", spaceAfter = 0, lineSpacing } = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox", name, position: box, fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  // A paragraph is a string, an array of runs, or { runs, spaceAfter } for its own spacing.
  const toRuns = (p) => (Array.isArray(p) ? p : [p])
    .map((part) => (typeof part === "string" ? run(part, { size, color, bold }) : part));
  // Size, colour and weight live on each run; setting them on the shape would override the runs.
  shape.text.style = {
    typeface: FONT, alignment: align, verticalAlignment: valign,
    insets: 0, wrap: "square", autoFit: "none", ...(lineSpacing ? { lineSpacing } : {}),
  };
  shape.text = paragraphs.map((p) => (p && !Array.isArray(p) && typeof p === "object" && "runs" in p
    ? { runs: toRuns(p.runs), spaceAfter: p.spaceAfter ?? spaceAfter }
    : { runs: toRuns(p), spaceAfter }));
  return shape;
}

function addTitle(slide, title) {
  addText(slide, "title", { left: MARGIN, top: 40, width: CONTENT_WIDTH, height: 64 }, [title],
    { size: SIZE.title, color: COLOR.navy, bold: true });
}

function addSource(slide, textValue) {
  addText(slide, "source-note", { left: MARGIN, top: 670, width: CONTENT_WIDTH, height: 30 }, [textValue],
    { size: SIZE.source, color: COLOR.muted, valign: "bottom" });
}

function setNotes(slide, { speaker, seconds, spoken, sources }) {
  const lines = [`Speaker: ${speaker}. Time: ${seconds} seconds.`, ...spoken, ...sources.map((s) => `Source: ${s}`)];
  slide.speakerNotes.text = lines;
  notesBySlide.push({ speaker, seconds, spoken, sources });
  deckText.push(...lines);
}

const bold = (textValue, color = COLOR.navy, size = SIZE.body) => run(textValue, { bold: true, color, size });
const numberWord = (n) => ["zero", "one", "two", "three", "four", "five", "six"][n] ?? String(n);
const sentence = (value) => value.charAt(0).toUpperCase() + value.slice(1).toLowerCase();
const DOI = "https://doi.org/10.24432/C5CG6D";
const DATA_SOURCE = "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. " +
  `${DOI}. Licence CC BY 4.0.`;

// --------------------------------------------------------------------------
// Slides

const presentation = Presentation.create({ slideSize: SLIDE });

function newSlide(background = COLOR.white) {
  const slide = presentation.slides.add();
  slide.background.fill = background;
  return slide;
}

// 1. Cover
{
  const slide = newSlide(COLOR.navy);
  addText(slide, "cover-title", { left: 96, top: 250, width: 1088, height: 84 }, ["Customers Like You"],
    { size: SIZE.cover, color: COLOR.white, bold: true });
  addText(slide, "cover-subtitle", { left: 96, top: 350, width: 1088, height: 44 },
    ["Product recommendations and customer groups"], { size: SIZE.coverSub, color: COLOR.coverTeal });
  addText(slide, "cover-course", { left: 96, top: 420, width: 1088, height: 32 }, ["DS5020 Mini-Project 1"],
    { size: 22, color: COLOR.white });
  setNotes(slide, {
    speaker: DATA_ENGINEER, seconds: 20,
    spoken: [
      "Sarah Okafor, our Head of E-Commerce, asked for two things: an also-bought box on product pages, " +
      "refreshed overnight and better than best sellers, and one simple picture of our customer groups for " +
      "marketing. We cover the data, the model, the groups and the risks.",
    ],
    sources: [DATA_SOURCE],
  });
}

// 2. Data and test design
{
  const slide = newSlide();
  addTitle(slide, "Training data and a novel-purchase test");
  const colWidth = 520;
  const top = 150;
  addText(slide, "train-heading", { left: MARGIN, top, width: colWidth, height: 36 },
    [`Training: ${f.train_start} to ${f.train_end}`], { size: 25, color: COLOR.teal, bold: true });
  addText(slide, "train-body", { left: MARGIN, top: top + 56, width: colWidth, height: 420 }, [
    [bold(f.raw_rows), run(" invoice lines, "), bold(f.clean_rows), run(" after cleaning")],
    [bold(f.train_rows), run(" training lines before the cutoff")],
    [bold(`${f.customers} x ${f.products}`), run(" customers by products")],
    [bold(f.sparsity), run(" of cells are empty")],
    [run("Cell value: log(1 + total quantity)")],
    [run("Products need at least "), bold(f.min_buyers), run(" training buyers")],
  ], { spaceAfter: 1400 });
  const right = MARGIN + colWidth + 112;
  addText(slide, "test-heading", { left: right, top, width: colWidth, height: 36 },
    [`Test: ${f.test_start} to ${f.test_end}`], { size: 25, color: COLOR.teal, bold: true });
  addText(slide, "test-body", { left: right, top: top + 56, width: colWidth, height: 420 }, [
    [run("Last two months, held out before any fitting")],
    [bold(f.test_rows), run(" invoice lines of new-to-customer products")],
    [bold(f.excluded_repeat_rows), run(" repeat-purchase lines left out")],
    [bold(`${f.eligible_customers} of ${f.total_test_customers}`), run(` customers scored (${f.eligible_customer_coverage_pct})`)],
    [bold(`${f.test_products_in_catalogue} of ${f.test_products}`), run(` test products in catalogue (${f.product_coverage_pct})`)],
    [bold(f.pair_coverage_pct), run(" of new customer-product pairs scored")],
  ], { spaceAfter: 1400 });
  addSource(slide, `Source: UCI Online Retail II (Chen, 2012), ${DOI}, CC BY 4.0. ` +
    "Counts from reports/matrix_report.json and reports/results.json.");
  setNotes(slide, {
    speaker: DATA_ENGINEER, seconds: 70,
    spoken: [
      `Our data is the UCI Online Retail II log, ${f.raw_rows} invoice lines from December 2009 to December ` +
      "2011. I removed cancellations, non-positive quantities and prices, non-product codes and lines without a " +
      `customer ID, leaving ${f.clean_rows}.`,
      `The test is the last two months, from ${f.test_start}. Targets are products new to that customer: ` +
      `${f.test_rows} invoice lines, with ${f.excluded_repeat_rows} repeat-purchase lines left out. The ` +
      `${f.train_rows} earlier lines are training data.`,
      "The matrix uses training lines only: log of one plus total quantity, and products with at least " +
      `${f.min_buyers} training buyers. That gives ${f.customers} customers by ${f.products} products, ` +
      `${f.sparsity} empty. Coverage on the right counts customers, distinct products and new pairs separately.`,
      "I built the original cleaning, split, matrix and first exploratory SVD. The completed package reruns " +
      "these steps from the raw file with python build_matrix.py.",
    ],
    sources: [DATA_SOURCE, "reports/cleaning_log.csv, reports/split_log.csv, reports/matrix_report.json."],
  });
}

// 3. Choice of k and compression
{
  const slide = newSlide();
  addTitle(slide, `Rank ${f.k} and the compression tradeoff`);
  const chartBox = { left: MARGIN - 8, top: 130, width: 680, height: 500 };
  const k = facts.chosen_k;
  const factorTarget = typeof facts.targets?.factor_mb_max === "number" &&
    facts.svd.factor_mb <= facts.targets.factor_mb_max ? facts.targets.factor_mb_max : undefined;
  const chart = slide.charts.add("line", {
    position: chartBox,
    categories: facts.curve.map((row) => String(row.k)),
    series: [{
      name: "Relative reconstruction error",
      // Workbooks hold 15 significant digits; the labels show one decimal of a percent.
      values: facts.curve.map((row) => Number(row.relative_error.toPrecision(15))),
      smooth: false,
      line: { style: "solid", fill: COLOR.teal, width: 3 },
      marker: { symbol: "circle", size: 9, fill: COLOR.teal },
      points: facts.curve.map((row, idx) => ({ row, idx })).filter(({ row }) => row.k === k)
        .map(({ idx }) => ({ idx, fill: COLOR.navy })),
    }],
    hasLegend: false,
    lineOptions: { smooth: false },
    xAxis: {
      title: { text: "Components kept (k)", textStyle: { fontSize: 18, fill: COLOR.muted } },
      textStyle: { fontSize: 18, fill: COLOR.ink },
      line: { style: "solid", fill: COLOR.rule, width: 1 },
    },
    yAxis: {
      title: { text: "Relative reconstruction error", textStyle: { fontSize: 18, fill: COLOR.muted } },
      min: 0, max: 1, majorUnit: 0.25, numberFormatCode: "0%",
      textStyle: { fontSize: 18, fill: COLOR.ink },
      majorGridlines: { style: "solid", fill: "#E5E7EB", width: 1 },
    },
    dataLabels: {
      showValue: true, position: "top", numberFormatCode: "0.0%",
      textStyle: { fontSize: 18, fill: COLOR.ink, bold: true },
    },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  deckText.push("Components kept (k)", "Relative reconstruction error", ...facts.curve.map((r) => String(r.k)));

  const left = 792;
  const width = SLIDE.width - MARGIN - left;
  addText(slide, "k-explanation", { left, top: 150, width, height: 470 }, [
    [run(`k = ${f.k} was fixed from this training curve before any test result`)],
    [run("Relative reconstruction error: "), bold(f.svd_error)],
    { runs: [run("Dense table: "), bold(f.dense_mb)], spaceAfter: 600 },
    { runs: [run("Sparse matrix storage: "), bold(f.sparse_mb)], spaceAfter: 600 },
    { runs: [run("Model factors: "), bold(f.svd_factor_mb)], spaceAfter: 600 },
    [run(`${f.dense_to_factor_ratio} times smaller than dense, ` +
      (o.factors_exceed_csr ? "but larger than sparse" : "and smaller than sparse"))],
  ], { spaceAfter: 1600 });
  addSource(slide, "Source: separate TruncatedSVD fits on the training matrix, reports/svd_curve.csv. " +
    "Error = ||A - A_k|| / ||A|| (Frobenius). MB = 1,000,000 bytes.");
  setNotes(slide, {
    speaker: ML_ENGINEER, seconds: 55,
    spoken: [
      "I chose Truncated SVD because it works on the sparse matrix directly. The chart shows relative reconstruction error on training data: the Frobenius norm " +
      "of the matrix minus its rank-k approximation, divided by the norm of the matrix. It describes how roughly " +
      "we approximate the log-transformed purchase patterns, not a share of purchases kept. It falls from " +
      `${f.curve_10} at 10 components to ${f[`curve_${k}`]} at ${k}.`,
      `k = ${f.k} came from Alex's original exploratory training curve and was fixed before any test result` +
      (factorTarget ? `. It keeps the factors under our ${factorTarget} MB target: ` : ": ") +
      `${f.svd_factor_mb}, ${f.dense_to_factor_ratio} times less than the dense table, ` +
      (o.factors_exceed_csr ? `but more than the ${f.sparse_mb} sparse matrix storage.`
        : `and less than the ${f.sparse_mb} sparse matrix storage.`),
    ],
    sources: ["reports/svd_curve.csv, reports/results.json (matrix and svd sections)."],
  });
}

// 4. PCA vs SVD comparison (ML5)
{
  const slide = newSlide();
  addTitle(slide, "PCA and SVD on the same matrix");
  const pcaDetail = f.pca_error ? `${f.pca_variance} explained, ${f.pca_error} error`
    : `${f.pca_variance} variance explained`;
  const hits = (model) => `${f[`${model}_hit`]} (${f[`${model}_hits`]}/${f.eligible_customers})`;
  const svdDetail = `${f.svd_error} relative error`;
  const rows = [
    ["Measure", "PCA", "Truncated SVD"],
    ["Components (k)", f.k, f.k],
    ["Fit time", f.pca_fit, f.svd_fit],
    ["Peak memory during fit", f.pca_peak, f.svd_peak],
    ["Stored factors", `${f.pca_factor_mb} (with mean)`, f.svd_factor_mb],
    ["Variance explained or error", pcaDetail, svdDetail],
    ["Hit rate at 10", hits("pca"), hits("svd")],
    ["Best-seller hit rate at 10", hits("baseline"), hits("baseline")],
    ["Component 1 meaning", facts.pca.component_1_short, facts.svd.component_1_short],
  ];
  rows.flat().forEach((cell) => deckText.push(String(cell)));
  const rowHeights = [50, 48, 48, 48, 48, 48, 48, 48, 48];
  const table = slide.tables.add({
    rows: rows.length, columns: 3, left: MARGIN, top: 122, width: CONTENT_WIDTH,
    height: rowHeights.reduce((a, b) => a + b, 0),
    columnWidths: [340, 406, 406],
    values: rows.map((row) => row.map(String)),
  });
  rowHeights.forEach((height, index) => { table.rows[index].height = height; });
  table.cells.block({ row: 0, column: 0, rowCount: rows.length, columnCount: 3 }).assign({
    textStyle: { fontSize: 22, typeface: FONT, color: COLOR.ink },
    margins: { left: 14, right: 14, top: 4, bottom: 4 },
    anchor: "middle",
    borders: { bottom: { style: "solid", fill: COLOR.rule, width: 1 } },
  });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 3 }).assign({
    fill: COLOR.navy, textStyle: { fontSize: 22, typeface: FONT, color: COLOR.white, bold: true },
  });
  table.cells.block({ row: 1, column: 0, rowCount: rows.length - 1, columnCount: 1 }).assign({
    textStyle: { fontSize: 22, typeface: FONT, color: COLOR.navy, bold: true },
  });
  const pcaVerdict = o.svd_pca_difference_not_established
    ? `SVD minus PCA: ${o.svd_vs_pca_points.toFixed(1)} points (95% interval ${f.svd_vs_pca_ci}), no established difference`
    : (o.svd_beats_pca ? `SVD beats PCA by ${o.svd_vs_pca_points.toFixed(1)} points`
      : `PCA matches or beats SVD by ${Math.abs(o.svd_vs_pca_points).toFixed(1)} points`);
  const baseVerdict = o.svd_beats_baseline && facts.pca.hits > facts.baseline.hits ? "Both beat best sellers. "
    : (o.svd_beats_baseline ? "SVD beats best sellers. " : "SVD does not beat best sellers. ");
  addText(slide, "comparison-takeaway", { left: MARGIN, top: 628, width: CONTENT_WIDTH, height: 34 },
    [[run(baseVerdict + pcaVerdict + ".", { color: COLOR.navy })]]);
  addSource(slide, `Source: reports/comparison.csv. Hit rates use the same ${f.eligible_customers} ` +
    "test customers with history and at least one new-to-them catalogue product.");
  const versusBaseline = o.svd_beats_baseline
    ? `SVD beat best sellers by ${o.svd_vs_baseline_points.toFixed(1)} points` +
      (f.svd_vs_baseline_ci ? `, paired 95% interval ${f.svd_vs_baseline_ci}.` : ".")
    : `SVD did not beat best sellers: it was ${Math.abs(o.svd_vs_baseline_points).toFixed(1)} points lower.`;
  const versusPca = o.svd_pca_difference_not_established
    ? ` SVD and PCA are ${o.svd_vs_pca_points.toFixed(1)} points apart and the interval includes zero, so this ` +
      "test does not establish a difference."
    : (o.svd_beats_pca ? ` SVD also beat PCA by ${o.svd_vs_pca_points.toFixed(1)} points.`
      : ` PCA did as well or better, by ${Math.abs(o.svd_vs_pca_points).toFixed(1)} points.`);
  const timingKnown = /median of 5 runs/i.test(facts.methodology?.timing ?? "") &&
    /1 ms/.test(facts.methodology?.memory_sampling ?? "") && /fresh process/i.test(facts.methodology?.memory_sampling ?? "");
  const method = timingKnown
    ? "Fit time is the median of five single-thread runs. Peak memory covers a whole fresh process, sampled " +
      "every millisecond."
    : [facts.methodology?.timing, facts.methodology?.memory_sampling].filter(Boolean).join(". ");
  const headline = (model) => facts[model].component_1_description.split(":")[0].trim();
  const componentNote = headline("svd") === headline("pca")
    ? `In both, component one means ${headline("svd").toLowerCase()}, correlation ${f.svd_corr} and ` +
      `${f.pca_corr} with log total quantity.`
    : `SVD's component one: ${facts.svd.component_1_short}. PCA's component one: ${facts.pca.component_1_short}.`;
  if (!method) fail("results.json environment needs timing and memory_sampling descriptions for slide 4 notes");
  setNotes(slide, {
    speaker: ML_ENGINEER, seconds: 70,
    spoken: [
      `I ran PCA on the same matrix with the same ${f.k} components and seed. It centres implicitly, so it ` +
      "describes differences from the average customer.",
      `All hit rates use the same ${f.eligible_customers} customers. A hit means at least one product they ` +
      "had never bought before is in their top 10, excluding what they already bought. " +
      `SVD scored ${f.svd_hit}, PCA ${f.pca_hit} and best sellers ${f.baseline_hit}. ${versusBaseline}${versusPca}`,
      `${method} PCA's size includes its mean vector.`,
      componentNote,
    ],
    sources: [`reports/comparison.csv, reports/results.json. Timings measured on ${f.machine}.`],
  });
}

// 5. Customer map (ML6)
{
  const slide = newSlide();
  addTitle(slide, "Customer groups for marketing");
  const box = { left: MARGIN - 16, top: 118, width: 800, height: 540 };
  const ratio = Math.min(box.width / facts.customer_map.width, box.height / facts.customer_map.height);
  const imageSize = { width: Math.round(facts.customer_map.width * ratio),
    height: Math.round(facts.customer_map.height * ratio) };
  const image = slide.images.add({
    blob: mapBytes.buffer.slice(mapBytes.byteOffset, mapBytes.byteOffset + mapBytes.byteLength),
    contentType: "image/png",
    alt: facts.customer_map_display.alt,
    fit: "contain",
    position: { left: box.left, top: box.top, ...imageSize },
  });
  image.lockAspectRatio = true;

  const left = box.left + imageSize.width + 36;
  const width = SLIDE.width - MARGIN - left;
  const paragraphs = [];
  for (const segment of facts.segments) {
    paragraphs.push({ runs: [bold(segment.name, COLOR.navy, 21)], spaceAfter: 300 });
    const uk = segment.uk_share == null ? "" : `UK ${Math.round(segment.uk_share * 100)}%. `;
    paragraphs.push({ runs: [run(`${uk}Example: ${sentence(segment.example_product)}`,
      { size: SIZE.small, color: COLOR.muted })], spaceAfter: 1500 });
  }
  addText(slide, "segment-list", { left, top: 134, width, height: 520 }, paragraphs);
  addSource(slide, `Source: reports/segments.csv and ${facts.customer_map.path}, training data only. ` +
    "UK = share of the group's customers.");
  const segmentation = facts.methodology?.segmentation ?? "";
  const restarts = segmentation.match(/n_init=(\d+)/)?.[1];
  // segment_profiles.json records k-means on the standardised (log10 radius, angle) of components 1-2.
  const polar = /k-means/i.test(segmentation) && /standardi[sz]ed/i.test(segmentation) &&
    /log10 radius, angle/.test(segmentation) && /random_state=42/.test(segmentation);
  if (!polar) fail(`Unrecognised segmentation method in segment_profiles.json: ${segmentation}`);
  const display = facts.customer_map_display ?? {};
  if (!display.description) fail("facts have no verified customer map description");
  setNotes(slide, {
    speaker: ML_ENGINEER, seconds: 50,
    spoken: [
      `${display.description} k-means grouped customers on these two values, standardised, into ` +
      `${numberWord(facts.segments.length)} groups, seed 42` + (restarts ? `, ${restarts} restarts. ` : ". ") +
      "Training data only, and only two of the 100 components. Component one tracks buying volume, so size " +
      "partly reflects order size.",
      "Names come from each group's common and distinctive products, with one example each on the right." +
      (o.uk_dominates_every_segment && f.uk_min
        ? ` All groups are mainly UK-based, ${f.uk_min} to ${f.uk_max} of customers, and some have a larger overseas share.`
        : "") +
      " Marketing should test them.",
    ],
    sources: [`reports/segments.csv, reports/segment_profiles.json, ${facts.customer_map.path}.`],
  });
}

// 6. Decision and next steps
{
  const slide = newSlide();
  addTitle(slide, "Pilot and coverage");
  const colWidth = 520;
  const top = 150;
  addText(slide, "result-heading", { left: MARGIN, top, width: colWidth, height: 36 },
    ["What the test showed"], { size: 25, color: COLOR.teal, bold: true });
  const verdict = o.svd_beats_baseline
    ? [run("SVD top 10: "), bold(f.svd_hit), run(" of customers had a hit, against "), bold(f.baseline_hit),
      run(" for best sellers")]
    : [run("SVD top 10: "), bold(f.svd_hit), run(", below best sellers at "), bold(f.baseline_hit)];
  const SHORT_WEIGHTING = { count: "raw count", raw: "raw count", raw_count: "raw count", quantity: "raw count",
    binary: "1/0", log: "log", log1p: "log" };
  const shortWeighting = (row) => SHORT_WEIGHTING[row.weighting.toLowerCase()] ?? row.label;
  const sensitivity = facts.sensitivity.map((row) => `${shortWeighting(row)} ${(row.hit_rate_at_10 * 100).toFixed(1)}%`);
  addText(slide, "result-body", { left: MARGIN, top: top + 56, width: colWidth, height: 420 }, [
    verdict,
    [run(`From the prepared table, fit and top 10 for all ${f.products} products: `), bold(f.nightly_total)],
    [run(`Purchase value: ${sensitivity.join(", ")}. Log was fixed before testing.`)],
  ], { spaceAfter: 1600 });
  const right = MARGIN + colWidth + 112;
  addText(slide, "next-heading", { left: right, top, width: colWidth, height: 36 },
    ["Next steps"], { size: 25, color: COLOR.teal, bold: true });
  addText(slide, "next-body", { left: right, top: top + 56, width: colWidth, height: 420 }, [
    [run("A/B pilot on product pages against best sellers")],
    [run(`Best sellers for new products and those with under ${f.min_buyers} buyers`)],
    [run("Personalised lists only for customers with history")],
    [run("Marketing checks the group names before campaigns")],
    [run("Track coverage: "), bold(f.pair_coverage_pct), run(" of new customer-product pairs scored")],
  ], { spaceAfter: 1600 });
  addSource(slide, `Source: reports/results.json, reports/sensitivity.csv. Times measured on ${f.machine}. ` +
    "An offline hit rate does not prove extra sales.");
  const sens = facts.sensitivity.map((row) => `${shortWeighting(row)} ${(row.hit_rate_at_10 * 100).toFixed(1)}%`);
  const sensSentence = `By purchase value: ${sens.slice(0, -1).join(", ")} and ${sens.at(-1)}.`;
  const best = facts.sensitivity.find((row) => row.weighting === o.best_sensitivity_weighting);
  const bestNote = best && !/^log/i.test(best.weighting)
    ? ` ${sentence(shortWeighting(best))} scored slightly higher, but log was fixed before testing, so we kept it.`
    : " Log, which we fixed before testing, also scored best.";
  setNotes(slide, {
    speaker: ML_ENGINEER, seconds: 35,
    spoken: [
      (o.svd_beats_baseline
        ? `We recommend a pilot: SVD beat best sellers, ${f.svd_hit} against ${f.baseline_hit}.`
        : `The model did not beat best sellers, ${f.svd_hit} against ${f.baseline_hit}, so we keep best sellers for now.`) +
      ` From the prepared purchase table, the fit and all-product neighbour lists took ${f.nightly_total}; ` +
      `raw-data cleaning is extra. ${sensSentence}${bestNote}`,
      "Product pages can serve new customers; personalised lists need history, and new products need best " +
      "sellers. Alex would own the data refresh, and I the refit and coverage checks.",
    ],
    sources: ["reports/results.json (nightly and sensitivity sections), reports/sensitivity.csv."],
  });
}

// --------------------------------------------------------------------------
// Checks on the authored content

const countWords = (value) => (value.match(/[\w'’()./+%-]+/g) ?? []).length;
const spokenWords = notesBySlide.reduce((total, n) => total + countWords(n.spoken.join(" ")), 0);
const seconds = notesBySlide.reduce((total, n) => total + n.seconds, 0);
const problems = [];
const warnings = [...(facts.warnings ?? [])];
if (presentation.slides.items.length !== 6) problems.push("deck must have exactly six slides");
if (seconds !== 300) problems.push(`slide timings add up to ${seconds} s, not 300 s`);
if (spokenWords < NOTES_WORDS.hardMin || spokenWords > NOTES_WORDS.hardMax) {
  problems.push(`speaker notes have ${spokenWords} spoken words (need about ${NOTES_WORDS.min}-${NOTES_WORDS.max}); ` +
    `by slide: ${notesBySlide.map((n) => countWords(n.spoken.join(" "))).join(", ")}`);
} else if (spokenWords < NOTES_WORDS.min || spokenWords > NOTES_WORDS.max) {
  warnings.push(`speaker notes have ${spokenWords} spoken words; target ${NOTES_WORDS.min}-${NOTES_WORDS.max}`);
}
const placeholder = deckText.find((t) => /\b(TODO|TBD|XXX|placeholder|undefined|NaN)\b/i.test(t));
if (placeholder) problems.push(`placeholder or missing value in deck text: ${placeholder.slice(0, 80)}`);

const now = new Date();
const pad = (value) => String(value).padStart(2, "0");
const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
  `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
const runDir = path.join(ROOT, "tmp/artifacts/slides", `run-${stamp}`);
await fs.mkdir(runDir, { recursive: true });

const textFile = path.join(runDir, "deck-text.txt");
await fs.writeFile(textFile, deckText.join("\n") + "\n", "utf8");
const scan = spawnSync(RUNTIME_PYTHON, [path.join(ROOT, "scripts/create_submission.py"), "scan", "--input", textFile],
  { cwd: ROOT, encoding: "utf8" });
let privacy;
try { privacy = JSON.parse(scan.stdout); } catch { privacy = { error: scan.stderr || scan.stdout }; }
if (scan.status !== 0) problems.push(`privacy scan failed: ${JSON.stringify(privacy)}`);

if (problems.length) {
  await fs.writeFile(path.join(runDir, "problems.json"), JSON.stringify({ problems, warnings }, null, 2) + "\n");
  fail(`deck content failed checks (nothing exported):\n  ${problems.join("\n  ")}`);
}

// --------------------------------------------------------------------------
// Draft, finalize, render

const draftPath = path.join(runDir, "draft.pptx");
await (await PresentationFile.exportPptx(presentation)).save(draftPath);

// Draft-only runs still go through the finalizer, into a private subdirectory of this run.
const canonical = args["draft-only"] ? path.join(runDir, "final", "presentation.pptx") : path.resolve(args.output);
await fs.mkdir(path.dirname(canonical), { recursive: true });
const exists = async (file) => fs.lstat(file).then(() => true, () => false);
const stamped = (file) => file.replace(/\.pptx$/i, `-${stamp}.pptx`);
const canonicalExists = await exists(canonical);
const finalPath = canonicalExists ? stamped(canonical) : canonical;
if (await exists(finalPath)) fail(`${path.relative(ROOT, finalPath)} already exists`);

const rootEntriesBefore = new Set(await fs.readdir(ROOT));
const tableOwners = [4];
let result;
try {
  result = await finalizePresentation({
    workspaceDir: ROOT,
    candidatePath: draftPath,
    finalPath,
    pythonExecutable: RUNTIME_PYTHON,
    integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
    layoutArgs: [
      "--expected-slide-size-emu", SLIDE_EMU,
      "--validate-bullet-geometry",
      "--validate-heading-fit",
      "--cover-role", "cover",
      "--approved-image-sha256", facts.customer_map.sha256,
      ...tableOwners.flatMap((n) => ["--require-native-table-slide", String(n)]),
    ],
    explicitTotalSlideCount: 6,
    requiredNativeTableOwnerSlides: tableOwners,
    requiredNativeChartOwnerSlides: [3],
    materializeLiteralChartWorkbooks: true,
    fontPolicy: { basis: "design", families: [FONT] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(runDir, `${path.basename(finalPath)}.validation.json`),
  });
} finally {
  // The finalizer stages chart workbooks in the workspace root; keep them private with this run.
  for (const entry of await fs.readdir(ROOT)) {
    if (!rootEntriesBefore.has(entry) && entry.startsWith(".chart-data-")) {
      await fs.rename(path.join(ROOT, entry), path.join(runDir, entry.slice(1)));
    }
  }
}

const layoutWarnings = result.presentationLayout?.warnings ?? [];
const finalBytes = await fs.readFile(finalPath);
if (sha256(finalBytes) !== result.finalSha256) fail("final deck bytes differ from the validation receipt");

// Render every slide of the finalized file, not the in-memory deck.
const renderDir = path.join(runDir, "render");
await fs.mkdir(renderDir, { recursive: true });
const imported = await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renders = [];
const smallText = [];
for (const [index, slide] of imported.slides.items.entries()) {
  const stem = `slide-${index + 1}`;
  const png = await imported.export({ slide, format: "png", scale });
  await fs.writeFile(path.join(renderDir, `${stem}.png`), Buffer.from(await png.arrayBuffer()));
  const layout = await (await slide.export({ format: "layout" })).text();
  await fs.writeFile(path.join(renderDir, `${stem}.layout.json`), layout);
  renders.push(path.relative(ROOT, path.join(renderDir, `${stem}.png`)));
  for (const match of layout.matchAll(/"fontSize":\s*([0-9.]+)/g)) {
    const size = Number(match[1]);
    if (size < SIZE.small && size !== SIZE.source && !smallText.includes(`${index + 1}:${size}`)) {
      smallText.push(`${index + 1}:${size}`);
    }
  }
}
const montage = await imported.export({ format: "webp", montage: true });
await fs.writeFile(path.join(renderDir, "montage.webp"), Buffer.from(await montage.arrayBuffer()));
if (imported.slides.items.length !== 6) fail("finalized deck does not have six slides");
if (smallText.length) warnings.push(`text below ${SIZE.small}px (slide:size): ${smallText.join(", ")}`);

let published = finalPath;
if (canonicalExists && args.replace) {
  await fs.rename(finalPath, canonical);
  if (sha256(await fs.readFile(canonical)) !== result.finalSha256) fail("replaced deck differs from validated bytes");
  published = canonical;
}

const report = {
  final: path.relative(ROOT, published),
  finalSha256: result.finalSha256,
  receipt: path.relative(ROOT, result.receiptPath),
  replacedCanonical: canonicalExists && args.replace,
  slides: imported.slides.items.length,
  font: FONT,
  spokenNotesWords: spokenWords,
  notesSeconds: seconds,
  notesWordsBySlide: notesBySlide.map((n) => countWords(n.spoken.join(" "))),
  privacy,
  layoutWarnings,
  warnings,
  renders,
  montage: path.relative(ROOT, path.join(renderDir, "montage.webp")),
  facts: path.relative(ROOT, path.resolve(args.facts)),
};
await fs.writeFile(path.join(runDir, "run-report.json"), JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify(report, null, 2));
