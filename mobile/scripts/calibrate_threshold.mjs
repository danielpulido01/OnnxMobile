// Calibra el umbral de confianza de classify() con el pipeline completo en Node, sobre la
// misma muestra de 2000 audios de VALIDACIÓN que make_reference.py (semilla 42).
// Uso (desde mobile/): node scripts/calibrate_threshold.mjs [ruta a speech_commands_v0.02]
import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import ort from "onnxruntime-node";

import { argmax, LABELS, softmax } from "../src/voice/labels.ts";
import { audioToImage, fixLength, N_FRAMES, N_MELS, SAMPLE_RATE } from "../src/voice/preprocess.ts";
import { readWav } from "./wav.mjs";

const SCRIPTS = dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = join(SCRIPTS, "..", "assets", "model");
const DATASET = process.argv[2] ?? "C:\\Users\\Sebas30\\Desktop\\datasets\\speech_commands_v0.02";
const OUTPUT = join(tmpdir(), "onnxmobile_calibration.json");
const UNKNOWN = LABELS.indexOf("unknown");
const THRESHOLDS = [...Array.from({ length: 14 }, (_, i) => 0.3 + 0.05 * i), 0.99];

const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));
const melFilters = readJson(join(MODEL_DIR, "mel_filters.json")).data;
const hannWindow = readJson(join(MODEL_DIR, "hann_window.json")).data;
const preprocessing = readJson(join(MODEL_DIR, "preprocessing.json"));
if (preprocessing.classes.join() !== LABELS.join()) throw new Error("LABELS no coincide con preprocessing.json");

// La muestra la elige numpy (default_rng(42).choice) con las funciones de make_reference.py,
// para que sea exactamente la misma. Solo se lee validation_list.txt.
function sampleFromPython() {
  const code = [
    "import json, sys",
    "from pathlib import Path",
    "import numpy as np",
    "from make_reference import ACCURACY_SAMPLES, SEED, read_validation_list",
    "paths = read_validation_list(Path(sys.argv[1]))",
    "index = sorted(np.random.default_rng(SEED).choice(len(paths), size=ACCURACY_SAMPLES, replace=False))",
    "print(json.dumps([paths[i] for i in index]))",
  ].join("\n");
  const python = join(SCRIPTS, ".venv", "Scripts", "python.exe");
  const result = spawnSync(python, ["-c", code, DATASET], { cwd: SCRIPTS, encoding: "utf8", maxBuffer: 1 << 24 });
  if (result.status !== 0) throw new Error(`No se pudo obtener la muestra desde Python:\n${result.stderr}`);
  return JSON.parse(result.stdout);
}

const sample = sampleFromPython();
const session = await ort.InferenceSession.create(join(MODEL_DIR, "model.onnx"));
const records = [];
let skipped = 0;
for (const relative of sample) {
  const { samples, sampleRate } = readWav(join(DATASET, relative));
  if (sampleRate !== SAMPLE_RATE) throw new Error(`${relative}: ${sampleRate} Hz`);
  // Mismo filtro que is_usable() de make_reference.py: vacío o silencio digital.
  if (samples.length === 0 || samples.every((v) => v === 0)) {
    skipped++;
    continue;
  }
  const image = audioToImage(fixLength(samples), melFilters, hannWindow);
  const logits = (await session.run({ input: new ort.Tensor("float32", image, [1, 1, N_MELS, N_FRAMES]) })).logits.data;
  const label = preprocessing.label_to_id[preprocessing.source_to_label[relative.split("/")[0]]];
  records.push({ relative_path: relative, label, probabilities: softmax(logits) });
}
writeFileSync(OUTPUT, JSON.stringify({ classes: LABELS, records }));

const commands = records.filter((r) => r.label !== UNKNOWN);
const unknowns = records.filter((r) => r.label === UNKNOWN);
const correct = records.filter((r) => argmax(r.probabilities) === r.label).length;
console.log(`muestra: ${records.length} audios de validación (${skipped} descartados) | ` +
  `comandos reales: ${commands.length} | unknown: ${unknowns.length}`);
console.log(`exactitud argmax (sin umbral): ${(correct / records.length).toFixed(4)} (${correct}/${records.length})`);
console.log(`probabilidades guardadas en: ${OUTPUT}\n`);

// Para un comando real, a cada umbral le pasa exactamente una de estas cuatro cosas.
const pct = (n, total) => ((100 * n) / total).toFixed(1).padStart(6);
const rows = THRESHOLDS.map((threshold) => {
  let ok = 0, wrong = 0, lowConfidence = 0, asUnknown = 0, falseTriggers = 0;
  for (const r of commands) {
    const predicted = argmax(r.probabilities);
    if (predicted === UNKNOWN) asUnknown++;
    else if (r.probabilities[predicted] < threshold) lowConfidence++;
    else if (predicted === r.label) ok++;
    else wrong++;
  }
  for (const r of unknowns) {
    const predicted = argmax(r.probabilities);
    if (predicted !== UNKNOWN && r.probabilities[predicted] >= threshold) falseTriggers++;
  }
  return { threshold, ok, wrong, lowConfidence, asUnknown, falseTriggers };
});

console.log("umbral | acept. correctos | rechazo baja conf. | acept. otro cmd | pred. unknown || falsos disparos unknown");
for (const r of rows) {
  const n = commands.length;
  console.log(`  ${r.threshold.toFixed(2)} |      ${pct(r.ok, n)} % |        ${pct(r.lowConfidence, n)} % |     ` +
    `${pct(r.wrong, n)} % |    ${pct(r.asUnknown, n)} % ||          ${pct(r.falseTriggers, unknowns.length)} %`);
}

// Sugerencia: minimizar comandos perdidos (todo lo que no es "aceptado correcto") + falsos disparos,
// ambos en %, con el mismo peso. En empate gana el umbral más bajo.
const cost = (r) => (100 * (commands.length - r.ok)) / commands.length + (100 * r.falseTriggers) / unknowns.length;
const best = rows.reduce((a, b) => (cost(b) < cost(a) - 1e-9 ? b : a));
console.log(`\nsugerencia: umbral ${best.threshold.toFixed(2)} ` +
  `(comandos perdidos ${(100 - (100 * best.ok) / commands.length).toFixed(1)} % + ` +
  `falsos disparos ${((100 * best.falseTriggers) / unknowns.length).toFixed(1)} % = ${cost(best).toFixed(1)})`);
