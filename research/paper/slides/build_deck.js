// Builds the 12-slide project deck. Every number is taken from the paper (research/paper/ieee/main.tex) and its results files.
//   NODE_PATH=<dir with pptxgenjs, react-icons, react, react-dom, sharp> node research/paper/slides/build_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const HERE = __dirname;
const IMG = (n) => path.join(HERE, "img", n);
const OUT = path.join(HERE, "malaria_danet_audit.pptx");

const THEME = {
  name: "Giemsa",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F1A2B", lt1: "FFFFFF", dk2: "3D2466", lt2: "F3EEF9",
    accent1: "6B3FA0", accent2: "EB6834", accent3: "2A78D6", accent4: "1BAF7A", accent5: "C2185B", accent6: "6E6A78",
    hlink: "2A78D6", folHlink: "6B3FA0",
  },
};
const HEX = { dark: "3D2466", violet: "6B3FA0", lav: "F3EEF9", lav2: "E3D9F2", orange: "EB6834", blue: "2A78D6", ink: "1F1A2B", muted: "6E6A78", white: "FFFFFF" };

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
  pres.author = "Agrapujya Lashkari, Deepanshu Mehta";
  pres.title = "High NIH accuracy does not predict transfer";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  // ---------- layouts ----------
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: HEX.dark },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.75, w: 8.8, h: 1.4, fontSize: 40, bold: true, color: C.background1, valign: "bottom", fontFace: THEME.headFontFace }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.2, w: 8.8, h: 0.9, fontSize: 18, color: C.background2, valign: "top" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: HEX.white },
    margin: [0.5, 0.5, 0.5, 0.5],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9.0, h: 0.68, fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", fontFace: THEME.headFontFace }, text: "" } },
      { text: { text: "DANet malaria audit  ·  Lashkari & Mehta", options: { x: 0.5, y: 5.22, w: 5, h: 0.3, fontSize: 9, color: HEX.muted, margin: 0 } } },
    ],
    slideNumber: { x: 9.1, y: 5.22, w: 0.4, h: 0.3, fontSize: 9, color: HEX.muted, align: "right" },
  });
  pres.defineSlideMaster({
    title: "DARK_CONTENT",
    background: { color: HEX.dark },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9.0, h: 0.68, fontSize: 30, bold: true, color: C.background1, valign: "middle", align: "left", fontFace: THEME.headFontFace }, text: "" } },
    ],
    slideNumber: { x: 9.1, y: 5.22, w: 0.4, h: 0.3, fontSize: 9, color: HEX.lav2, align: "right" },
  });

  // ---------- helpers ----------
  const card = (s, x, y, w, h, name, fill = HEX.lav) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: name });
  const iconDot = async (s, Comp, x, y, d, name, bg = HEX.violet) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg }, objectName: name + " circle" });
    s.addImage({ data: await icon(Comp, "FFFFFF"), x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52, objectName: name + " icon", altText: name });
  };
  const text = (s, t, o) => s.addText(t, { isTextBox: true, margin: 0, color: HEX.ink, fontSize: 14, valign: "top", ...o });
  const stat = (s, big, label, x, y, w, color = HEX.violet, name = "stat") => {
    text(s, big, { x, y, w, h: 0.62, fontSize: 36, bold: true, color, fontFace: THEME.headFontFace, objectName: name + " value" });
    text(s, label, { x, y: y + 0.64, w, h: 0.62, fontSize: 12, color: HEX.ink, objectName: name + " label" });
  };

  // ===== 1. Title =====
  pres.addSection({ title: "Context" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Context" });
  s.addText("High NIH accuracy does not predict transfer", { placeholder: "title" });
  s.addText("Reproducing and auditing DANet for malaria cell classification, from NIH to a one-shot external test", { placeholder: "body" });
  s.addImage({ path: IMG("title_cells.png"), x: 0.6, y: 3.35, w: 5.6, h: 0.9, objectName: "Cell examples", altText: "Five red-cell crops: NIH parasitized and uninfected, MP-IDB, BBBC041 JPG and PNG" });
  text(s, "NIH  ·  MP-IDB  ·  BBBC041", { x: 0.6, y: 4.3, w: 5.6, h: 0.3, fontSize: 11, color: HEX.lav2, objectName: "Cell sources" });
  text(s, [{ text: "Agrapujya Lashkari  ·  Deepanshu Mehta", options: { bold: true, breakLine: true } },
           { text: "CV research-reproduction project, Newton School of Technology, Rishihood University" }],
    { x: 0.6, y: 4.75, w: 8.8, h: 0.6, fontSize: 12, color: HEX.white, objectName: "Authors" });
  s.addNotes("This project started as a course reproduction of DANet on the NIH malaria dataset. Our one pre-registered change failed, and that failure led us to ask what the 97% score actually measures. The talk follows that path: problem, baseline, hypothesis, then the audit and a one-shot external test.");

  // ===== 2. Problem =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Context" });
  s.addText("The problem: a 98% score that is hard to read", { placeholder: "title" });
  card(s, 0.5, 1.15, 3.1, 3.6, "Headline card");
  text(s, "97.95%", { x: 0.75, y: 1.4, w: 2.7, h: 0.85, fontSize: 48, bold: true, color: HEX.violet, fontFace: THEME.headFontFace, objectName: "DANet accuracy" });
  text(s, "DANet's reported accuracy on the NIH thin-smear cell dataset, with 2.3 M parameters", { x: 0.75, y: 2.3, w: 2.7, h: 0.8, fontSize: 13, objectName: "DANet accuracy label" });
  text(s, "53.2%", { x: 0.75, y: 3.2, w: 2.7, h: 0.6, fontSize: 30, bold: true, color: HEX.orange, fontFace: THEME.headFontFace, objectName: "MalariaNet accuracy" });
  text(s, "MalariaNet on BBBC041 without slide leakage (specificity 18%)", { x: 0.75, y: 3.8, w: 2.7, h: 0.7, fontSize: 13, objectName: "MalariaNet label" });
  const issues = [
    [fa.FaUsers, "Patient leakage", "Random cell splits put every test patient in the training set too."],
    [fa.FaAdjust, "A simple cue", "The parasite is usually the darkest object in a pale, curated crop."],
    [fa.FaRandom, "One checkpoint, one dataset", "Equal test scores can hide very different behaviour on new slides."],
  ];
  for (let i = 0; i < issues.length; i++) {
    const y = 1.15 + i * 1.0;
    await iconDot(s, issues[i][0], 3.95, y + 0.08, 0.62, issues[i][1]);
    text(s, issues[i][1], { x: 4.75, y, w: 4.75, h: 0.36, fontSize: 16, bold: true, color: HEX.dark, objectName: issues[i][1] + " title" });
    text(s, issues[i][2], { x: 4.75, y: y + 0.38, w: 4.75, h: 0.5, fontSize: 14, objectName: issues[i][1] + " text" });
  }
  text(s, "Question: what does DANet's NIH score measure, and does it transfer to new slides?", { x: 3.95, y: 4.3, w: 5.55, h: 0.5, fontSize: 14, italic: true, color: HEX.violet, objectName: "Research question" });
  s.addNotes("NIH has 27,558 segmented cells from 200 patients. Lightweight networks report 97 to 99 percent on it. Three reasons to doubt what that means: patients appear in both train and test under random splits, the crops are curated so the parasite is usually the darkest thing, and a single checkpoint on a single dataset can hide underspecification. MalariaNet's leakage-free result on BBBC041, 53 percent accuracy with 18 percent specificity, shows the risk.");

  // ===== 3. Baseline =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Context" });
  s.addText("Baseline: DANet reproduced within its own range", { placeholder: "title" });
  card(s, 0.5, 1.15, 2.9, 2.35, "Model card");
  await iconDot(s, fa.FaMicroscope, 0.7, 1.33, 0.55, "DANet");
  text(s, "DANet-S", { x: 1.38, y: 1.4, w: 1.9, h: 0.4, fontSize: 18, bold: true, color: HEX.dark, objectName: "Model name" });
  text(s, [
    { text: "2.31 M parameters, 0.83 GFLOPs", options: { bullet: true, breakLine: true } },
    { text: "Dilated attention + InstanceNorm", options: { bullet: true, breakLine: true } },
    { text: "RMSprop, one-cycle, max LR 0.001", options: { bullet: true } },
  ], { x: 0.7, y: 2.0, w: 2.6, h: 1.4, fontSize: 13, paraSpaceAfter: 4, objectName: "Model facts" });
  const th = { bold: true, color: HEX.white, fill: { color: HEX.violet }, align: "center" };
  const row = (cells, bold) => cells.map((c, i) => ({ text: c, options: { align: i ? "center" : "left", bold: !!bold } }));
  s.addTable([
    [{ text: "NIH random split", options: { ...th, align: "left" } }, { text: "ACC", options: th }, { text: "SE", options: th }, { text: "SP", options: th }, { text: "AUC", options: th }],
    row(["Paper, single split", "97.95", "98.07", "97.87", "0.98"]),
    row(["Paper, 5-fold mean", "97.25", "97.20", "–", "–"]),
    row(["Ours (seed 42)", "97.02", "96.30", "97.75", "0.9945"], true),
  ], { x: 3.7, y: 1.15, w: 5.8, colW: [2.2, 0.9, 0.9, 0.9, 0.9], fontSize: 13, color: HEX.ink, border: { type: "solid", pt: 0.5, color: HEX.lav2 }, rowH: 0.42, fill: { color: HEX.white }, objectName: "Reproduction table" });
  text(s, "Gap: −0.93 pp accuracy, inside the paper's own 5-fold range (96.97–97.42). It sits in sensitivity: 102 missed parasitized cells.", { x: 3.7, y: 2.95, w: 5.8, h: 0.6, fontSize: 13, color: HEX.ink, objectName: "Gap note" });
  text(s, "The released code needed three repairs before it ran", { x: 0.5, y: 3.75, w: 9, h: 0.35, fontSize: 15, bold: true, color: HEX.dark, objectName: "Repairs heading" });
  const fixes = [[fa.FaBug, "Failed on import"], [fa.FaLayerGroup, "32-way output for 2 classes"], [fa.FaSlidersH, "Paper's LR (0.01) did not train"]];
  for (let i = 0; i < 3; i++) {
    const x = 0.5 + i * 3.05;
    card(s, x, 4.2, 2.85, 0.75, "Repair " + (i + 1));
    await iconDot(s, fixes[i][0], x + 0.15, 4.31, 0.52, fixes[i][1], HEX.orange);
    text(s, fixes[i][1], { x: x + 0.8, y: 4.2, w: 1.95, h: 0.75, fontSize: 13, valign: "middle", objectName: "Repair text " + (i + 1) });
  }
  s.addNotes("We reproduced DANet-S on the paper's own random split. We reached 97.02 percent accuracy and AUC 0.9945, about one point below the 97.95 headline but inside the paper's five-fold range. The gap is in sensitivity. Three repairs were needed: the code failed on import, the output layer had 32 classes, and the paper's learning rate of 0.01 did not train, so we used the only consistent setting that did.");

  // ===== 4. Hypothesis =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Context" });
  s.addText("Our hypothesis (M3): colour augmentation", { placeholder: "title" });
  card(s, 0.5, 1.15, 4.1, 3.75, "Hypothesis card");
  text(s, "If we add stain-colour augmentation, DANet will do better on unseen patients.", { x: 0.75, y: 1.35, w: 3.65, h: 0.85, fontSize: 16, bold: true, color: HEX.dark, objectName: "Hypothesis statement" });
  text(s, [
    { text: "Why: ", options: { bold: true } }, { text: "patient identity explains 74–90% of a cell's mean colour, so each slide's stain is a fingerprint a network can use.", options: { breakLine: true } },
    { text: "Change: ", options: { bold: true } }, { text: "ColorJitter (brightness 0.1, contrast 0.1, saturation 0.5, hue 0.1), nothing else.", options: { breakLine: true } },
    { text: "Written: ", options: { bold: true } }, { text: "2026-10-05, before any code." },
  ], { x: 0.75, y: 2.3, w: 3.65, h: 2.4, fontSize: 13, paraSpaceAfter: 8, objectName: "Hypothesis detail" });
  // number line: H minus B1, balanced accuracy
  const NX = 5.0, NW = 4.4, lo = -0.5, hi = 1.5, X = (v) => NX + ((v - lo) / (hi - lo)) * NW;
  text(s, "H − B1, balanced accuracy on unseen patients (pp)", { x: 5.0, y: 1.2, w: 4.5, h: 0.35, fontSize: 13, bold: true, color: HEX.dark, objectName: "Number line title" });
  s.addShape(pres.shapes.RECTANGLE, { x: X(0.3), y: 1.75, w: X(1.0) - X(0.3), h: 0.42, fill: { color: HEX.lav2 }, line: { color: HEX.lav2 }, objectName: "Predicted range" });
  text(s, "Predicted +0.3 to +1.0", { x: X(0.3), y: 1.75, w: X(1.0) - X(0.3), h: 0.42, fontSize: 11, align: "center", valign: "middle", color: HEX.dark, objectName: "Predicted label" });
  s.addShape(pres.shapes.LINE, { x: X(-0.22), y: 2.55, w: X(0.14) - X(-0.22), h: 0, line: { color: HEX.orange, width: 3 }, objectName: "Measured CI" });
  s.addShape(pres.shapes.OVAL, { x: X(-0.04) - 0.09, y: 2.46, w: 0.18, h: 0.18, fill: { color: HEX.orange }, line: { color: HEX.white, width: 1 }, objectName: "Measured point" });
  text(s, "Measured −0.04 [−0.22, +0.14]", { x: X(0.2), y: 2.4, w: 2.4, h: 0.3, fontSize: 11, color: HEX.orange, bold: true, objectName: "Measured label" });
  s.addShape(pres.shapes.LINE, { x: NX, y: 2.95, w: NW, h: 0, line: { color: HEX.muted, width: 1 }, objectName: "Axis" });
  s.addShape(pres.shapes.LINE, { x: X(0), y: 1.65, w: 0, h: 1.3, line: { color: HEX.muted, width: 0.75, dashType: "dash" }, objectName: "Zero line" });
  [-0.5, 0, 0.5, 1.0, 1.5].forEach((v) => text(s, (v > 0 ? "+" : "") + v.toFixed(1), { x: X(v) - 0.3, y: 3.0, w: 0.6, h: 0.25, fontSize: 10, align: "center", color: HEX.muted, objectName: "Tick " + v }));
  card(s, 5.0, 3.45, 4.5, 1.45, "Verdict card", "FCEAE2");
  text(s, "Falsified", { x: 5.2, y: 3.55, w: 4.1, h: 0.45, fontSize: 22, bold: true, color: HEX.orange, fontFace: THEME.headFontFace, objectName: "Verdict" });
  text(s, "Leakage was smaller than predicted too: unseen patients cost only 0.23 pp, not 1–2.5. InstanceNorm already removes much of a global stain shift.", { x: 5.2, y: 4.0, w: 4.1, h: 0.85, fontSize: 12, objectName: "Verdict detail" });
  s.addNotes("Milestone 3 hypothesis, written before any code: colour augmentation removes the stain fingerprint and helps on unseen patients. We predicted plus 0.3 to plus 1.0 points of balanced accuracy. The paired result was minus 0.04, with a confidence interval from minus 0.22 to plus 0.14, which excludes the whole predicted range. The control on the leaky split behaved as predicted. Our own 'why we might be wrong' note named the reason: InstanceNorm. This failure is what started the audit.");

  // ===== 5. Study design =====
  pres.addSection({ title: "Audit on NIH" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Audit on NIH" });
  s.addText("From one change to a full audit", { placeholder: "title" });
  const steps = [
    [fa.FaDatabase, "NIH", "27,558 cells, 200 patients"],
    [fa.FaColumns, "3 splits", "leaky, patient, session"],
    [fa.FaCogs, "Models", "DANet, ResNet-18, probes, dark-spot rule"],
    [fa.FaFlask, "MP-IDB", "development, 63 checkpoints"],
    [fa.FaLock, "Freeze", "4 candidates, hashes, outcome rule"],
    [fa.FaBullseye, "BBBC041", "scored once, 79,722 cells"],
  ];
  for (let i = 0; i < steps.length; i++) {
    const x = 0.5 + i * 1.53;
    const fill = i === 5 ? "FCEAE2" : HEX.lav;
    card(s, x, 1.15, 1.4, 1.75, "Step " + (i + 1), fill);
    await iconDot(s, steps[i][0], x + 0.43, 1.27, 0.54, steps[i][1], i === 5 ? HEX.orange : HEX.violet);
    text(s, steps[i][1], { x: x + 0.08, y: 1.88, w: 1.24, h: 0.3, fontSize: 14, bold: true, align: "center", color: HEX.dark, objectName: "Step title " + (i + 1) });
    text(s, steps[i][2], { x: x + 0.08, y: 2.2, w: 1.24, h: 0.65, fontSize: 11, align: "center", objectName: "Step text " + (i + 1) });
  }
  text(s, "Every experiment had a dated prediction before it ran. Failed predictions stay on record.", { x: 0.5, y: 3.0, w: 9, h: 0.3, fontSize: 13, italic: true, color: HEX.violet, objectName: "Design note" });
  s.addImage({ path: IMG("fig_examples.png"), x: 1.1, y: 3.4, w: 6.65, h: 1.9, objectName: "Example crops", altText: "Random parasitized and uninfected crops from NIH, MP-IDB and BBBC041 JPG and PNG groups" });
  text(s, "NIH and MP-IDB: P. falciparum. BBBC041: P. vivax, two acquisition groups with very different stain.", { x: 7.95, y: 3.55, w: 1.55, h: 1.5, fontSize: 11, color: HEX.muted, objectName: "Example caption" });
  s.addNotes("After the hypothesis failed, we registered a series of experiments. NIH with three splits: leaky, patient-grouped, and session-grouped by imaging date. References: ResNet-18, frozen foundation-model probes, and a one-measurement dark-spot rule with no training. MP-IDB was used for development. Then we froze four candidates with hashes and an outcome rule, and scored BBBC041 exactly once. The images show how different BBBC041's PNG group looks.");

  // ===== 6. NIH =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Audit on NIH" });
  s.addText("On NIH, every effect lands within one point", { placeholder: "title" });
  s.addImage({ path: IMG("fig_forest.png"), x: 0.5, y: 1.1, w: 5.35, h: 3.86, objectName: "Forest plot", altText: "Paired differences in NIH balanced accuracy with 95% CIs, against a 0.29 pp minimum detectable effect" });
  stat(s, "+0.26 pp", "Patient leakage (leaky − grouped), below the 0.29 pp detectable effect", 6.25, 1.1, 3.25, HEX.violet, "Leakage");
  stat(s, "95.6%", "Balanced accuracy of one darkness measurement, with no training", 6.25, 2.42, 3.25, HEX.orange, "Rule");
  stat(s, "+1.16 pp", "DANet's real margin over that rule [0.90, 1.45], with 888 vs 1,208 errors", 6.25, 3.74, 3.25, HEX.blue, "Margin");
  s.addNotes("Every comparison on NIH is within about one point. Patient leakage: plus 0.26, below our minimum detectable effect of 0.29 from an A/A test. Session leakage: plus 0.24, also below it, though one repeat cannot rule it out. ResNet-18 and frozen probes come within 0.1 to 0.7. A single darkness measurement reaches 95.6 percent balanced accuracy and AUC 0.981 with no training. DANet beats it by 1.16 points, and that margin is real. So NIH mostly measures a strong simple cue.");

  // ===== 7. Explanations =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Audit on NIH" });
  s.addText("Explanations: the darkest pixel beats Grad-CAM", { placeholder: "title" });
  s.addImage({ path: IMG("fig_explain.png"), x: 0.5, y: 1.15, w: 5.9, h: 2.89, objectName: "Explanation plot", altText: "Hits on parasite for darkest pixel, Grad-CAM and crop centre, five folds per DANet variant" });
  text(s, "Share of correctly detected MP-IDB cells where each pointer lands on the expert parasite mask. Dots are folds; bars are means.", { x: 0.5, y: 4.15, w: 5.9, h: 0.6, fontSize: 12, color: HEX.muted, objectName: "Explanation caption" });
  stat(s, "15 / 15", "Checkpoints where the no-model darkest pixel hit parasites more often", 6.8, 1.15, 2.7, HEX.orange, "Wins");
  stat(s, "+27.3 pp", "Darkest pixel minus Grad-CAM for DANet [15.8, 38.0]", 6.8, 2.45, 2.7, HEX.violet, "Gap");
  stat(s, "8–16%", "Grad-CAM hit rate on 3 of 5 folds for one variant, 90% on the others", 6.8, 3.75, 2.7, HEX.blue, "Instability");
  s.addNotes("We asked how often an explanation points at the real parasite. On all 15 checkpoints, the darkest smoothed pixel, chosen with no model at all, hit the parasite more often than Grad-CAM's peak. Grad-CAM was also unstable: one variant hit about 90 percent on two folds and 8 to 16 percent on three. Our earlier single-checkpoint Grad-CAM result was not representative. These maps show dependence on a dark object, not learned morphology.");

  // ===== 8. Seeds =====
  pres.addSection({ title: "External test" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "External test" });
  s.addText("Same data, new seed: NIH flat, external spread", { placeholder: "title" });
  s.addImage({ path: IMG("fig_seeds.png"), x: 0.5, y: 1.1, w: 5.75, h: 3.75, objectName: "Seed plot", altText: "Ten checkpoints differing only by seed: NIH balanced accuracy near 97, MP-IDB from 59.6 to 85.7" });
  card(s, 6.6, 1.1, 2.9, 1.15, "NIH SD card");
  text(s, "0.18", { x: 6.8, y: 1.15, w: 1.3, h: 0.7, fontSize: 40, bold: true, color: HEX.violet, fontFace: THEME.headFontFace, objectName: "NIH SD" });
  text(s, "seed SD on NIH test folds", { x: 8.0, y: 1.25, w: 1.4, h: 0.9, fontSize: 12, valign: "middle", objectName: "NIH SD label" });
  card(s, 6.6, 2.45, 2.9, 1.15, "MP-IDB SD card", "FCEAE2");
  text(s, "6.63", { x: 6.8, y: 2.5, w: 1.3, h: 0.7, fontSize: 40, bold: true, color: HEX.orange, fontFace: THEME.headFontFace, objectName: "MP-IDB SD" });
  text(s, "seed SD on MP-IDB [4.48, 12.71]", { x: 8.0, y: 2.6, w: 1.4, h: 0.9, fontSize: 12, valign: "middle", objectName: "MP-IDB SD label" });
  text(s, [
    { text: "Two seeds on one fold: 0.15 pp apart on NIH, 13.7 pp apart on MP-IDB.", options: { bullet: true, breakLine: true } },
    { text: "Predicted SD 3–6: missed on the high side.", options: { bullet: true } },
  ], { x: 6.6, y: 3.8, w: 2.9, h: 1.1, fontSize: 13, paraSpaceAfter: 6, objectName: "Seed notes" });
  s.addNotes("Each fold changes both the training patients and the model, so we isolated the seed: two folds, five seeds each, same training patients. On NIH, their pooled within-fold SD is 0.18 points, and no fold spans more than 0.64. On MP-IDB, they span 59.6 to 85.7 percent, a pooled SD of 6.63. NIH could not warn us of this, so a single checkpoint is a poor unit of external evidence. This is underspecification in the sense of D'Amour et al.");

  // ===== 9. BBBC041 result =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "External test" });
  s.addText("One-shot test on BBBC041: trained model wins", { placeholder: "title" });
  const proto = ["Freeze 4 candidates; hash code and checkpoints", "Score both crop rules; commit hashes before any label", "Calculate once; never rescore", "Outcome must agree on both crop rules"];
  for (let i = 0; i < proto.length; i++) {
    const y = 1.15 + i * 0.66;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.05, w: 0.42, h: 0.42, fill: { color: HEX.violet }, line: { color: HEX.violet }, objectName: "Step dot " + (i + 1) });
    text(s, String(i + 1), { x: 0.5, y: y + 0.05, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: HEX.white, align: "center", valign: "middle", objectName: "Step number " + (i + 1) });
    text(s, proto[i], { x: 1.05, y, w: 2.85, h: 0.55, fontSize: 13, valign: "middle", objectName: "Protocol step " + (i + 1) });
  }
  card(s, 0.5, 3.85, 3.4, 1.05, "Prediction card", "FCEAE2");
  text(s, [{ text: "Our leading prediction: ", options: { bold: true } }, { text: "the rule holds (45%). We gave this outcome 15%." }],
    { x: 0.7, y: 3.95, w: 3.0, h: 0.85, fontSize: 13, valign: "middle", objectName: "Prediction note" });
  s.addChart(pres.charts.BAR, [
    { name: "Ellipse crops", labels: ["Dark-spot rule", "DANet E1 ensemble", "Colour-transfer ensemble"], values: [61.3, 87.9, 92.1] },
    { name: "Otsu crops", labels: ["Dark-spot rule", "DANet E1 ensemble", "Colour-transfer ensemble"], values: [64.4, 81.8, 89.2] },
  ], {
    x: 4.25, y: 1.1, w: 5.25, h: 3.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [HEX.violet, "B79FD9"], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.ink,
    dataLabelFontFace: "+mn-lt", showTitle: true, title: "Balanced accuracy at frozen NIH thresholds (%)", titleFontSize: 12, titleColor: HEX.dark, titleFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25, valAxisLabelFontSize: 10, catAxisLabelFontSize: 11, valAxisLabelColor: HEX.muted, catAxisLabelColor: HEX.ink,
    valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", valGridLine: { color: "E6E2EC", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.ink, objectName: "BBBC041 BACC chart",
  });
  text(s, "AUC, ellipse / Otsu: rule 0.909 / 0.906 · DANet 0.966 / 0.975 · colour-transfer 0.977 / 0.980. Colour-transfer minus rule: +30.8 [29.8, 31.9] and +24.8 [23.7, 25.7] pp BACC.", { x: 4.25, y: 4.45, w: 5.25, h: 0.6, fontSize: 11, color: HEX.muted, objectName: "AUC note" });
  s.addNotes("BBBC041 is P. vivax, 79,619 cells at a natural 3.05 percent prevalence, and no model or rule had seen it. Four candidates were frozen and hashed first; scores were committed before any label was read; metrics were calculated once. The colour-transfer ensemble beat the dark-spot rule under both crop rules, by 31 and 25 points of balanced accuracy, with CIs far above the margin. The registered outcome is 'trained model wins'. Our own leading prediction, that the rule would hold, was wrong.");

  // ===== 10. Why the rule failed =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "External test" });
  s.addText("The rule kept its ranking but lost its threshold", { placeholder: "title" });
  s.addImage({ path: IMG("fig_bbbc041.png"), x: 0.75, y: 1.05, w: 8.5, h: 3.58, objectName: "BBBC041 figure", altText: "ROC curves for both crop rules with frozen operating points, and dark-spot score distributions for NIH, BBBC041 JPG and PNG" });
  text(s, [{ text: "PNG group (93% of cells): ", options: { bold: true } }, { text: "82% of uninfected cells score above the NIH threshold, so specificity falls to 22.9%. " },
           { text: "JPG group: ", options: { bold: true } }, { text: "the same rule reaches 96.4% BACC." }],
    { x: 0.75, y: 4.68, w: 8.5, h: 0.5, fontSize: 13, objectName: "Threshold note" });
  s.addNotes("Panels a and b: the rule's ROC curve stays above AUC 0.90, so it still ranks cells. But its operating point, the dot, sits far to the right: it calls almost everything parasitized. Panel c shows why. The NIH threshold separates the classes on NIH and on BBBC041's JPG group, but in the PNG group, which is 93 percent of the cells, 82 percent of uninfected cells score above it. The same pattern appears in MalariaNet's published BBBC041 result. The colour-transfer ensemble's fixed threshold was within 0.2 points of its best case.");

  // ===== 11. Failure analysis =====
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "External test" });
  s.addText("Failure analysis: where each candidate breaks", { placeholder: "title" });
  s.addImage({ path: IMG("fig_confusion.png"), x: 0.8, y: 1.02, w: 8.4, h: 2.17, objectName: "Confusion matrices", altText: "Row-normalised confusion matrices: DANet on NIH, and three candidates on BBBC041" });
  s.addImage({ path: IMG("fig_failures.png"), x: 0.8, y: 3.25, w: 8.4, h: 1.57, objectName: "Failure gallery", altText: "Random BBBC041 errors: missed rings, false calls by the ensemble, PNG uninfected and leukocytes called by the rule" });
  text(s, "Rings: colour-transfer ensemble caught 75% (ellipse), 96% (Otsu).  Leukocytes flagged: rule 99%, DANet 98%, colour-transfer 23%.", { x: 0.8, y: 4.88, w: 8.4, h: 0.3, fontSize: 12, objectName: "Recall note" });
  s.addNotes("Top: confusion matrices. DANet on NIH is balanced. On BBBC041 the rule calls 77 percent of uninfected cells parasitized; the plain DANet ensemble 23 percent; the colour-transfer ensemble 8 percent, with 92 percent sensitivity. Bottom: errors drawn at random. Missed rings are faint dots in dark PNG cells. Leukocytes, which NIH never contains, are large dark objects that a darkness rule cannot reject; the colour-transfer ensemble rejected about three quarters of them.");

  // ===== 12. Conclusion =====
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "DARK_CONTENT", sectionTitle: "Conclusion" });
  s.addText("What we learned", { placeholder: "title" });
  text(s, "3 / 13", { x: 0.5, y: 1.2, w: 2.9, h: 0.9, fontSize: 54, bold: true, color: HEX.orange, fontFace: THEME.headFontFace, objectName: "Prediction score" });
  text(s, "main predictions fully supported. The misses taught us the most.", { x: 0.5, y: 2.15, w: 2.9, h: 0.9, fontSize: 14, color: HEX.white, objectName: "Prediction label" });
  const lessons = [
    [fa.FaAdjust, "NIH's 97% mostly measures a simple dark-spot cue; DANet adds a real 1.16 pp."],
    [fa.FaRandom, "A new seed alone moves external accuracy by SD 6.6; use ensembles, not one checkpoint."],
    [fa.FaPalette, "Colour transfer did nothing on NIH but was the only approach that transferred cleanly."],
    [fa.FaLock, "Only a frozen, untouched external test showed which approach works; MP-IDB misled us."],
  ];
  for (let i = 0; i < lessons.length; i++) {
    const y = 1.15 + i * 0.82;
    await iconDot(s, lessons[i][0], 3.8, y + 0.04, 0.52, "Lesson " + (i + 1), HEX.orange);
    text(s, lessons[i][1], { x: 4.5, y, w: 5.0, h: 0.62, fontSize: 14, color: HEX.white, valign: "middle", objectName: "Lesson text " + (i + 1) });
  }
  text(s, "Limits: no clinical claim (PPV 13–26% at 3% prevalence); one external dataset; species, stain and acquisition change together; session leakage tested on one repeat.", { x: 0.5, y: 4.55, w: 9, h: 0.55, fontSize: 12, color: HEX.lav2, objectName: "Limitations" });
  s.addNotes("Only four of our fifteen main predictions were fully supported, and the misses were the most useful part. NIH's 97 percent mostly reflects a simple cue. External accuracy depends on the seed and on the operating point, which NIH cannot show. The colour-transfer ensemble transferred best, though it also differs in size and architecture mix, so colour transfer alone is not isolated. And only a frozen one-shot test told us the truth; our development dataset pointed the other way. No result here supports clinical use.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
