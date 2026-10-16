// Compara src/voice/preprocess.ts con los espectrogramas de referencia de librosa.
// Uso (desde mobile/): node scripts/check_preprocess.mjs
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { audioToImage, fixLength, N_FRAMES, N_MELS, SAMPLE_RATE } from "../src/voice/preprocess.ts";
import { readWav } from "./wav.mjs";

const MOBILE = join(dirname(fileURLToPath(import.meta.url)), "..");
const REFERENCE = join(MOBILE, "scripts", "reference");
const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));
const melFilters = readJson(join(MOBILE, "assets", "model", "mel_filters.json")).data;
const hannWindow = readJson(join(MOBILE, "assets", "model", "hann_window.json")).data;

// .npy little-endian float32 en orden C. Con forma (64, 101, 1) el índice plano es
// m*101 + t, el mismo orden [banda][frame] que devuelve audioToImage.
function readNpy(path) {
  const buffer = readFileSync(path);
  if (buffer[0] !== 0x93 || buffer.toString("latin1", 1, 6) !== "NUMPY") throw new Error(`${path}: no es .npy`);
  const major = buffer[6];
  const headerLength = major === 1 ? buffer.readUInt16LE(8) : buffer.readUInt32LE(8);
  const start = (major === 1 ? 10 : 12) + headerLength;
  const header = buffer.toString("latin1", major === 1 ? 10 : 12, start);
  const shape = header.match(/'shape':\s*\(([^)]*)\)/)[1].split(",").filter((s) => s.trim()).map(Number);
  if (!header.includes("'descr': '<f4'") || !header.includes("'fortran_order': False")) {
    throw new Error(`${path}: se espera float32 little-endian en orden C`);
  }
  const count = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(count);
  for (let i = 0; i < count; i++) data[i] = buffer.readFloatLE(start + 4 * i);
  return { shape, data };
}

let globalMax = 0;
const names = readdirSync(join(REFERENCE, "wav")).filter((n) => n.endsWith(".wav")).sort();
for (const name of names) {
  const { samples, sampleRate } = readWav(join(REFERENCE, "wav", name));
  if (sampleRate !== SAMPLE_RATE) throw new Error(`${name}: ${sampleRate} Hz, se espera ${SAMPLE_RATE}`);
  const image = audioToImage(fixLength(samples), melFilters, hannWindow);
  const expected = readNpy(join(REFERENCE, "npy", name.replace(/\.wav$/, ".npy")));
  if (expected.shape.join("x") !== `${N_MELS}x${N_FRAMES}x1`) throw new Error(`${name}: forma ${expected.shape}`);
  let max = 0;
  for (let i = 0; i < image.length; i++) max = Math.max(max, Math.abs(image[i] - expected.data[i]));
  globalMax = Math.max(globalMax, max);
  console.log(`${name.padEnd(36)} muestras=${String(samples.length).padStart(5)}  error máx = ${max.toExponential(3)}`);
}
console.log(`audios: ${names.length} | error máximo global = ${globalMax.toExponential(3)}`);
