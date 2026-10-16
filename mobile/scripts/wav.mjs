// Lector de WAV PCM de 16 bits para los scripts de comprobación.
import { readFileSync } from "node:fs";

// Como soundfile: int16 / 32768 y promedio de canales.
export function readWav(path) {
  const buffer = readFileSync(path);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  const tag = (offset) => buffer.toString("ascii", offset, offset + 4);
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error(`${path}: no es WAV`);
  let format = null;
  for (let offset = 12; offset + 8 <= buffer.length; ) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      format = {
        code: view.getUint16(body, true),
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        bits: view.getUint16(body + 14, true),
      };
    } else if (id === "data") {
      if (!format || format.code !== 1 || format.bits !== 16) throw new Error(`${path}: se espera PCM de 16 bits`);
      const frames = Math.floor(size / (2 * format.channels));
      const mono = new Float32Array(frames);
      for (let i = 0; i < frames; i++) {
        let sum = 0;
        for (let c = 0; c < format.channels; c++) sum += view.getInt16(body + 2 * (i * format.channels + c), true) / 32768;
        mono[i] = sum / format.channels;
      }
      return { samples: mono, sampleRate: format.sampleRate };
    }
    offset = body + size + (size % 2); // los chunks se alinean a 2 bytes
  }
  throw new Error(`${path}: sin chunk data`);
}
