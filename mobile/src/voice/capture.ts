import type { AudioRecorder } from "react-native-audio-api";

import { fixLength, N_SAMPLES, SAMPLE_RATE } from "./preprocess";

// Se graba un poco más de 1 s para no depender de cuándo empieza a hablar el usuario.
export const RECORD_MS = 1500;
const BLOCK_LENGTH = SAMPLE_RATE / 10; // bloques de ~0.1 s

type AudioApi = typeof import("react-native-audio-api");
let audioApi: AudioApi | null = null;
let recorder: AudioRecorder | null = null;

// react-native-audio-api se carga con require() al primer uso: es código nativo que no existe
// en Expo Go, y un import arriba rompería la app en el iPhone.
function loadAudioApi(): AudioApi {
  if (!audioApi) {
    try {
      audioApi = require("react-native-audio-api") as AudioApi;
    } catch {
      throw new Error("Este binario no incluye react-native-audio-api (¿Expo Go?): usa el dev build");
    }
  }
  return audioApi;
}

function concat(chunks: Float32Array[]): Float32Array {
  const out = new Float32Array(chunks.reduce((n, chunk) => n + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

/**
 * Graba RECORD_MS con react-native-audio-api (Android e iOS, solo dev build) y entrega 16000 muestras
 * a 16 kHz en [-1, 1]: la ventana de 1 s con más energía. Se pide 16 kHz mono, pero se usa la
 * frecuencia que reporta cada bloque y, si no es 16000, se remuestrea con resampleLinear.
 */
export async function captureAudio(): Promise<Float32Array> {
  const { AudioManager, AudioRecorder } = loadAudioApi();
  if ((await AudioManager.requestRecordingPermissions()) !== "Granted") {
    throw new Error("Permiso de micrófono denegado");
  }
  // iOS: "measurement" desactiva el procesamiento de voz (AGC, etc.). En Android no tiene efecto.
  AudioManager.setAudioSessionOptions({ iosCategory: "record", iosMode: "measurement", iosOptions: [] });
  await AudioManager.setAudioSessionActivity(true);

  recorder ??= new AudioRecorder();
  const chunks: Float32Array[] = [];
  const blockRates: number[] = []; // frecuencia que reporta cada bloque
  const setup = recorder.onAudioReady(
    { sampleRate: SAMPLE_RATE, bufferLength: BLOCK_LENGTH, channelCount: 1 },
    ({ buffer }) => {
      blockRates.push(buffer.sampleRate);
      // Copia (slice): el bloque puede reutilizar memoria nativa. Si llega en estéreo, se promedia.
      const mono = buffer.getChannelData(0).slice();
      for (let c = 1; c < buffer.numberOfChannels; c++) {
        const channel = buffer.getChannelData(c);
        for (let i = 0; i < mono.length; i++) mono[i] += channel[i];
      }
      if (buffer.numberOfChannels > 1) for (let i = 0; i < mono.length; i++) mono[i] /= buffer.numberOfChannels;
      chunks.push(mono);
    },
  );
  try {
    if (setup.status === "error") throw new Error(`onAudioReady: ${setup.message}`);
    const started = await recorder.start();
    if (started.status === "error") throw new Error(`No se pudo grabar: ${started.message}`);
    await new Promise((resolve) => setTimeout(resolve, RECORD_MS));
    await recorder.stop();
  } finally {
    recorder.clearOnAudioReady();
    await AudioManager.setAudioSessionActivity(false);
  }

  if (chunks.length === 0) throw new Error("El micrófono no entregó audio");
  const rates = new Set(blockRates);
  const [rate] = rates;
  if (rates.size > 1) console.warn(`[capture] frecuencias distintas entre bloques: ${[...rates].join(", ")} Hz; se usa ${rate}`);
  const raw = concat(chunks);
  const resampled = resampleLinear(raw, rate, SAMPLE_RATE);
  const start = loudestStart(resampled);
  const window = loudestWindow(resampled);
  if (__DEV__) {
    let sumSquares = 0;
    for (const value of window) sumSquares += value * value;
    const seconds = (samples: number) => (samples / SAMPLE_RATE).toFixed(3);
    const end = Math.min(start + N_SAMPLES, resampled.length);
    console.log(`[capture] frecuencia por bloque (Hz): ${blockRates.join(", ")}`);
    console.log(
      `[capture] muestras capturadas=${raw.length} a ${rate} Hz (${(raw.length / rate).toFixed(2)} s), ` +
        (rate !== SAMPLE_RATE ? `remuestreo ${rate} -> ${SAMPLE_RATE} Hz: ${resampled.length} muestras` : "sin remuestreo"),
    );
    console.log(
      `[capture] ventana: muestras ${start}-${end} de ${resampled.length} (${seconds(start)}-${seconds(end)} s)` +
        (resampled.length < N_SAMPLES ? `, relleno con ${N_SAMPLES - resampled.length} ceros al final` : "") +
        `, pico=${peakOf(window).toFixed(4)}, RMS=${Math.sqrt(sumSquares / window.length).toFixed(4)}`,
    );
  }
  return window;
}

/** Máximo valor absoluto de las muestras (lo usa el filtro de silencio). */
export function peakOf(samples: Float32Array): number {
  let peak = 0;
  for (const value of samples) peak = Math.max(peak, Math.abs(value));
  return peak;
}

/** Remuestreo por interpolación lineal (sin filtro antialiasing; suficiente para pruebas). */
export function resampleLinear(samples: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return samples;
  const out = new Float32Array(Math.floor((samples.length * to) / from));
  for (let i = 0; i < out.length; i++) {
    const position = (i * from) / to;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    const fraction = position - left;
    out[i] = samples[left] * (1 - fraction) + samples[right] * fraction;
  }
  return out;
}

/** Ventana de 16000 muestras con más energía; si el audio es más corto, fixLength (ceros al final). */
export function loudestWindow(samples: Float32Array): Float32Array {
  if (samples.length <= N_SAMPLES) return fixLength(samples);
  const start = loudestStart(samples);
  return samples.slice(start, start + N_SAMPLES);
}

/** Inicio de la ventana de 16000 muestras con más energía (0 si el audio no es más largo). */
function loudestStart(samples: Float32Array): number {
  if (samples.length <= N_SAMPLES) return 0;
  let energy = 0;
  for (let i = 0; i < N_SAMPLES; i++) energy += samples[i] * samples[i];
  let best = energy;
  let bestStart = 0;
  for (let start = 1; start + N_SAMPLES <= samples.length; start++) {
    const leaving = samples[start - 1];
    const entering = samples[start + N_SAMPLES - 1];
    energy += entering * entering - leaving * leaving;
    if (energy > best) {
      best = energy;
      bestStart = start;
    }
  }
  return bestStart;
}
