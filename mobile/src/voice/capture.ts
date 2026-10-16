import {
  AudioQuality,
  IOSOutputFormat,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
  type RecordingOptions,
} from "expo-audio";
import { File } from "expo-file-system";
import { useCallback } from "react";

import { fixLength, N_SAMPLES, SAMPLE_RATE } from "./preprocess";

// Se graba un poco más de 1 s para no depender de cuándo empieza a hablar el usuario.
export const RECORD_MS = 1500;

// WAV mono PCM lineal de 16 bits a 16 kHz. Solo iOS: en Android expo-audio no ofrece
// WAV/PCM (solo AAC, AMR, etc.), así que ahí parseWav fallará con un error claro.
const RECORDING_OPTIONS: RecordingOptions = {
  extension: ".wav",
  sampleRate: SAMPLE_RATE,
  numberOfChannels: 1,
  bitRate: SAMPLE_RATE * 16,
  ios: {
    outputFormat: IOSOutputFormat.LINEARPCM,
    audioQuality: AudioQuality.MAX,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
  android: { outputFormat: "default", audioEncoder: "default" },
  web: {},
};

const PCM = 1;
const EXTENSIBLE = 0xfffe;

/** WAV PCM de 16 bits little-endian -> muestras en [-1, 1] (int16 / 32768, promedio de canales). */
export function parseWav(bytes: Uint8Array): { samples: Float32Array; sampleRate: number } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (offset: number) => String.fromCharCode(...bytes.subarray(offset, offset + 4));
  if (bytes.length < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("La grabación no es un WAV");
  let format: { code: number; channels: number; sampleRate: number; bits: number } | null = null;
  // Recorre los chunks: iOS puede poner otros (p. ej. "FLLR") antes de "data".
  for (let offset = 12; offset + 8 <= bytes.length; ) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      let code = view.getUint16(body, true);
      if (code === EXTENSIBLE && size >= 26) code = view.getUint16(body + 24, true); // subformato
      format = {
        code,
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        bits: view.getUint16(body + 14, true),
      };
    } else if (id === "data") {
      if (!format || format.code !== PCM || format.bits !== 16) {
        throw new Error(`WAV no soportado: formato ${format?.code}, ${format?.bits} bits (se espera PCM de 16)`);
      }
      const frames = Math.floor(Math.min(size, bytes.length - body) / (2 * format.channels));
      const samples = new Float32Array(frames);
      for (let i = 0; i < frames; i++) {
        let sum = 0;
        for (let c = 0; c < format.channels; c++) {
          sum += view.getInt16(body + 2 * (i * format.channels + c), true) / 32768;
        }
        samples[i] = sum / format.channels;
      }
      return { samples, sampleRate: format.sampleRate };
    }
    offset = body + size + (size % 2); // los chunks se alinean a 2 bytes
  }
  throw new Error("WAV sin chunk data");
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
  return samples.slice(bestStart, bestStart + N_SAMPLES);
}

/** Hook: devuelve capture(), que graba RECORD_MS y entrega 16000 muestras a 16 kHz en [-1, 1]. */
export function useVoiceCapture(): () => Promise<Float32Array> {
  const recorder = useAudioRecorder(RECORDING_OPTIONS);

  return useCallback(async () => {
    const permission = await requestRecordingPermissionsAsync();
    if (!permission.granted) throw new Error("Permiso de micrófono denegado");
    await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });

    await recorder.prepareToRecordAsync();
    recorder.record();
    await new Promise((resolve) => setTimeout(resolve, RECORD_MS));
    await recorder.stop();
    if (!recorder.uri) throw new Error("La grabación no generó archivo");

    const file = new File(recorder.uri);
    const bytes = await file.bytes();
    file.delete();
    const { samples, sampleRate } = parseWav(bytes);
    if (sampleRate !== SAMPLE_RATE) console.log(`[capture] grabado a ${sampleRate} Hz, remuestreando a ${SAMPLE_RATE}`);
    return loudestWindow(resampleLinear(samples, sampleRate, SAMPLE_RATE));
  }, [recorder]);
}
