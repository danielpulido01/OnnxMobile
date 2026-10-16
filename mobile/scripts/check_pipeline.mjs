// Pipeline completo en Node: WAV -> preprocess.ts -> model.onnx (onnxruntime-node) -> labels.ts,
// comparado con los logits de PyTorch/onnxruntime guardados en scripts/reference/reference.json.
// Uso (desde mobile/): node scripts/check_pipeline.mjs
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import ort from "onnxruntime-node";

import { classify, LABELS } from "../src/voice/labels.ts";
import { audioToImage, fixLength, N_FRAMES, N_MELS, SAMPLE_RATE } from "../src/voice/preprocess.ts";
import { readWav } from "./wav.mjs";

const MOBILE = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_DIR = join(MOBILE, "assets", "model");
const REFERENCE = join(MOBILE, "scripts", "reference");
const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));
const melFilters = readJson(join(MODEL_DIR, "mel_filters.json")).data;
const hannWindow = readJson(join(MODEL_DIR, "hann_window.json")).data;
const preprocessing = readJson(join(MODEL_DIR, "preprocessing.json"));
const reference = readJson(join(REFERENCE, "reference.json"));

// labels.ts debe tener exactamente el orden del modelo.
for (const [name, classes] of [["preprocessing.json", preprocessing.classes], ["reference.json", reference.classes]]) {
  if (classes.join() !== LABELS.join()) throw new Error(`LABELS no coincide con ${name}`);
}

const session = await ort.InferenceSession.create(join(MODEL_DIR, "model.onnx"));
const byWav = new Map(reference.items.map((item) => [item.wav.replace(/^wav\//, ""), item]));
const names = readdirSync(join(REFERENCE, "wav")).filter((n) => n.endsWith(".wav")).sort();

let globalMax = 0;
let matches = 0;
console.log(`onnxruntime-node ${ort.env.versions.node} | ${names.length} audios`);
for (const name of names) {
  const item = byWav.get(name);
  if (!item) throw new Error(`${name} no está en reference.json`);
  const { samples, sampleRate } = readWav(join(REFERENCE, "wav", name));
  if (sampleRate !== SAMPLE_RATE) throw new Error(`${name}: ${sampleRate} Hz, se espera ${SAMPLE_RATE}`);
  const image = audioToImage(fixLength(samples), melFilters, hannWindow);
  const input = new ort.Tensor("float32", image, [1, 1, N_MELS, N_FRAMES]);
  const logits = (await session.run({ input })).logits.data;

  let max = 0;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, Math.abs(logits[i] - item.logits[i]));
  globalMax = Math.max(globalMax, max);
  const prediction = classify(logits, 0);
  const same = prediction.label === item.predicted_label;
  matches += same;
  console.log(`${name.padEnd(32)} esperada=${item.label.padEnd(8)} predicha=${prediction.label.padEnd(8)} ` +
    `(ref ${item.predicted_label.padEnd(8)} ${same ? "igual" : "DISTINTA"}) ` +
    `conf=${prediction.confidence.toFixed(3)} dif. máx logits=${max.toExponential(3)}`);
}
console.log(`diferencia máxima global de logits = ${globalMax.toExponential(3)}`);
console.log(`clase igual a la referencia: ${matches}/${names.length}`);
