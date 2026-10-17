import { Directory, File, Paths } from "expo-file-system";

import { SAMPLE_RATE } from "./preprocess";

// Solo para depuración (__DEV__): guarda lo que entra al modelo para revisarlo en la PC con
// scripts/check_wav.mjs. En Android queda en files/captures/ de la app (sacar con adb run-as).
const FOLDER = "captures";
const NAME = /^capture_(\d+)\.wav$/;

/** Muestras en [-1, 1] -> WAV PCM de 16 bits, mono (inverso de readWav: x * 32768, recortado). */
export function encodeWav(samples: Float32Array, sampleRate = SAMPLE_RATE): Uint8Array {
  const bytes = new Uint8Array(44 + 2 * samples.length);
  const view = new DataView(bytes.buffer);
  const tag = (offset: number, text: string) => {
    for (let i = 0; i < 4; i++) bytes[offset + i] = text.charCodeAt(i);
  };
  tag(0, "RIFF");
  view.setUint32(4, 36 + 2 * samples.length, true);
  tag(8, "WAVE");
  tag(12, "fmt ");
  view.setUint32(16, 16, true); // tamaño del chunk fmt
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, 2 * sampleRate, true); // bytes por segundo
  view.setUint16(32, 2, true); // bytes por muestra
  view.setUint16(34, 16, true); // bits
  tag(36, "data");
  view.setUint32(40, 2 * samples.length, true);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.round(samples[i] * 32768);
    view.setInt16(44 + 2 * i, Math.max(-32768, Math.min(32767, value)), true);
  }
  return bytes;
}

/** Guarda capture_001.wav, capture_002.wav, ... y devuelve el nombre del archivo. */
export function saveCaptureWav(samples: Float32Array): string {
  const folder = new Directory(Paths.document, FOLDER);
  if (!folder.exists) folder.create();
  let last = 0;
  for (const entry of folder.list()) {
    const match = entry instanceof File ? NAME.exec(entry.name) : null;
    if (match) last = Math.max(last, Number(match[1]));
  }
  const name = `capture_${String(last + 1).padStart(3, "0")}.wav`;
  const file = new File(folder, name);
  file.create();
  file.write(encodeWav(samples));
  return name;
}
