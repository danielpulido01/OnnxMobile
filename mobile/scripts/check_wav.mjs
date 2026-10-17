// Clasifica en la PC un WAV guardado por la app (files/captures/capture_NNN.wav) con el mismo
// pipeline que el teléfono: readWav -> fixLength -> audioToImage -> model.onnx -> labels.ts.
// Uso (desde mobile/): node scripts/check_wav.mjs <archivo.wav> [más archivos...]
import { readFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import ort from "onnxruntime-node";

import { classify, LABELS } from "../src/voice/labels.ts";
import { audioToImage, fixLength, N_FRAMES, N_MELS, SAMPLE_RATE } from "../src/voice/preprocess.ts";
import { readWav } from "./wav.mjs";

const MODEL_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "assets", "model");
const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));
const melFilters = readJson(join(MODEL_DIR, "mel_filters.json")).data;
const hannWindow = readJson(join(MODEL_DIR, "hann_window.json")).data;

const paths = process.argv.slice(2);
if (paths.length === 0) {
  console.error("Uso: node scripts/check_wav.mjs <archivo.wav> [más archivos...]");
  process.exit(1);
}

const session = await ort.InferenceSession.create(join(MODEL_DIR, "model.onnx"));
for (const path of paths) {
  const { samples, sampleRate } = readWav(path);
  if (sampleRate !== SAMPLE_RATE) throw new Error(`${path}: ${sampleRate} Hz, se espera ${SAMPLE_RATE}`);
  const image = audioToImage(fixLength(samples), melFilters, hannWindow);
  const logits = (await session.run({ input: new ort.Tensor("float32", image, [1, 1, N_MELS, N_FRAMES]) })).logits.data;
  const { label, confidence, probabilities } = classify(logits, 0);

  console.log(`${basename(path)}: ${samples.length} muestras -> predicha ${label} (${confidence.toFixed(3)})`);
  const order = LABELS.map((_, i) => i).sort((a, b) => probabilities[b] - probabilities[a]);
  for (const i of order) {
    console.log(`  ${LABELS[i].padEnd(8)} ${probabilities[i].toFixed(4)} ${"#".repeat(Math.round(40 * probabilities[i]))}`);
  }
}
