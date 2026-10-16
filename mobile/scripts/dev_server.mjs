// SOLO DESARROLLO: clasifica en la PC (onnxruntime-node) el audio que graba la app en Expo Go,
// usando el mismo pipeline de la app: audioToImage -> model.onnx -> classify(logits, CONFIDENCE_THRESHOLD).
// El objetivo final es OnnxClassifier en Android, sin servidor. No exponer fuera de la red local.
// Uso (desde mobile/): node scripts/dev_server.mjs
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { networkInterfaces } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import ort from "onnxruntime-node";

import { CONFIDENCE_THRESHOLD } from "../src/voice/config.ts";
import { classify } from "../src/voice/labels.ts";
import { audioToImage, N_FRAMES, N_MELS, N_SAMPLES } from "../src/voice/preprocess.ts";

const PORT = 8787;
const MAX_BODY = 1 << 20; // 1 MiB: 16000 floats en JSON ocupan ~350 KB
const MODEL_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "assets", "model");
const readJson = (name) => JSON.parse(readFileSync(join(MODEL_DIR, name), "utf8"));
const melFilters = readJson("mel_filters.json").data;
const hannWindow = readJson("hann_window.json").data;
const session = await ort.InferenceSession.create(join(MODEL_DIR, "model.onnx"));

// Cuerpo: JSON {"samples": [16000 números]} o binario de 16000 float32 little-endian.
function parseSamples(body, contentType) {
  let samples;
  if (contentType.includes("application/json")) {
    samples = Float32Array.from(JSON.parse(body.toString("utf8")).samples ?? []);
  } else {
    samples = new Float32Array(body.length / 4);
    for (let i = 0; i < samples.length; i++) samples[i] = body.readFloatLE(4 * i);
  }
  if (samples.length !== N_SAMPLES) throw new Error(`Se esperan ${N_SAMPLES} muestras, llegaron ${samples.length}`);
  return samples;
}

async function handleClassify(samples) {
  const image = audioToImage(samples, melFilters, hannWindow);
  const output = await session.run({ input: new ort.Tensor("float32", image, [1, 1, N_MELS, N_FRAMES]) });
  const { command, label, confidence } = classify(output.logits.data, CONFIDENCE_THRESHOLD);
  return { command, label, confidence };
}

function send(response, status, value) {
  response.writeHead(status, { "Content-Type": "application/json" });
  response.end(JSON.stringify(value));
}

const server = createServer((request, response) => {
  if (request.method === "GET" && request.url === "/health") return send(response, 200, { ok: true });
  if (request.method !== "POST" || request.url !== "/classify") return send(response, 404, { error: "Use POST /classify" });

  const chunks = [];
  let size = 0;
  request.on("data", (chunk) => {
    size += chunk.length;
    if (size > MAX_BODY) request.destroy();
    else chunks.push(chunk);
  });
  request.on("end", async () => {
    try {
      const samples = parseSamples(Buffer.concat(chunks), request.headers["content-type"] ?? "");
      const result = await handleClassify(samples);
      let peak = 0;
      for (const v of samples) peak = Math.max(peak, Math.abs(v));
      console.log(`${new Date().toLocaleTimeString()}  ${request.socket.remoteAddress}  ` +
        `label=${result.label.padEnd(8)} conf=${result.confidence.toFixed(3)} ` +
        `command=${String(result.command).padEnd(7)} pico=${peak.toFixed(3)}`);
      send(response, 200, result);
    } catch (error) {
      console.error(`error: ${error.message}`);
      send(response, 400, { error: error.message });
    }
  });
});

server.listen(PORT, "0.0.0.0", () => {
  const addresses = Object.values(networkInterfaces()).flat()
    .filter((a) => a && a.family === "IPv4" && !a.internal).map((a) => a.address);
  console.log(`dev_server (SOLO DESARROLLO) escuchando en 0.0.0.0:${PORT} | umbral ${CONFIDENCE_THRESHOLD}`);
  for (const address of addresses) console.log(`  EXPO_PUBLIC_DEV_SERVER_URL=http://${address}:${PORT}`);
  if (addresses.length === 0) console.log("  No se encontró una IPv4 local: ¿está conectada la PC a la red?");
});
