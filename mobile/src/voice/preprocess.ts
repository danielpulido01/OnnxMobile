// Espectrograma log-Mel 64x101 igual a audio_to_image (scripts/make_reference.py,
// copia de la celda 13 de ai/notebooks/01_exploracion_speech_commands.ipynb).
// TypeScript puro: los cálculos van en float64 y la salida en float32.

export const SAMPLE_RATE = 16000;
export const N_SAMPLES = 16000;
export const N_FFT = 512;
export const HOP_LENGTH = 160;
export const N_MELS = 64;
export const N_BINS = N_FFT / 2 + 1; // 257
export const N_FRAMES = 1 + N_SAMPLES / HOP_LENGTH; // 101
export const TOP_DB = 80;
const AMIN = 1e-10;
const PAD = N_FFT / 2; // center=True, pad_mode="constant": 256 ceros por lado

// Tablas de la FFT radix-2: índices con bits invertidos y factores de giro.
const LOG2_N = Math.log2(N_FFT);
const BIT_REVERSED = new Uint16Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  let reversed = 0;
  for (let bit = 0; bit < LOG2_N; bit++) reversed |= ((i >> bit) & 1) << (LOG2_N - 1 - bit);
  BIT_REVERSED[i] = reversed;
}
const COS = new Float64Array(N_FFT / 2);
const SIN = new Float64Array(N_FFT / 2);
for (let k = 0; k < N_FFT / 2; k++) {
  COS[k] = Math.cos((2 * Math.PI * k) / N_FFT);
  SIN[k] = -Math.sin((2 * Math.PI * k) / N_FFT); // e^{-2πik/N}
}

// FFT compleja iterativa (Cooley-Tukey) in place sobre re/im de N_FFT puntos.
function fft(re: Float64Array, im: Float64Array): void {
  for (let i = 0; i < N_FFT; i++) {
    const j = BIT_REVERSED[i];
    if (j > i) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let size = 2; size <= N_FFT; size *= 2) {
    const half = size / 2;
    const step = N_FFT / size;
    for (let start = 0; start < N_FFT; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = COS[k * step];
        const wi = SIN[k * step];
        const a = start + k;
        const b = a + half;
        const tr = re[b] * wr - im[b] * wi;
        const ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

/**
 * Como load_fixed_audio: rellena con ceros al final o recorta al centro hasta 16000 muestras.
 */
export function fixLength(samples: Float32Array): Float32Array {
  if (samples.length > N_SAMPLES) {
    const start = Math.floor((samples.length - N_SAMPLES) / 2);
    return samples.slice(start, start + N_SAMPLES);
  }
  const fixed = new Float32Array(N_SAMPLES);
  fixed.set(samples);
  return fixed;
}

/**
 * Convierte 1 s de audio mono a 16 kHz en la entrada del modelo.
 *
 * @param signal 16000 muestras en [-1, 1].
 * @param melFilters `data` de assets/model/mel_filters.json (64*257, data[m*257 + k]).
 * @param window `data` de assets/model/hann_window.json (512 puntos).
 * @returns 64*101 valores en [0, 1], orden [banda][frame] (out[m*101 + t]),
 *   fila 0 = banda Mel más baja. Es el tensor NCHW [1, 1, 64, 101] sin batch ni canal.
 */
export function audioToImage(
  signal: Float32Array,
  melFilters: ArrayLike<number>,
  window: ArrayLike<number>,
): Float32Array {
  if (signal.length !== N_SAMPLES) throw new Error(`Se requieren ${N_SAMPLES} muestras`);
  if (melFilters.length !== N_MELS * N_BINS) throw new Error("mel_filters debe tener 64*257 valores");
  if (window.length !== N_FFT) throw new Error("La ventana debe tener 512 puntos");

  const padded = new Float64Array(N_SAMPLES + 2 * PAD);
  for (let i = 0; i < N_SAMPLES; i++) {
    if (!Number.isFinite(signal[i])) throw new Error("Audio con valores no finitos");
    padded[PAD + i] = signal[i];
  }

  // Potencia |X|^2 de los 257 bins por frame y luego proyección a 64 bandas Mel.
  const mel = new Float64Array(N_MELS * N_FRAMES);
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const power = new Float64Array(N_BINS);
  for (let t = 0; t < N_FRAMES; t++) {
    const offset = t * HOP_LENGTH;
    for (let n = 0; n < N_FFT; n++) {
      re[n] = padded[offset + n] * window[n];
      im[n] = 0;
    }
    fft(re, im);
    for (let k = 0; k < N_BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < N_MELS; m++) {
      let sum = 0;
      const row = m * N_BINS;
      for (let k = 0; k < N_BINS; k++) sum += melFilters[row + k] * power[k];
      mel[m * N_FRAMES + t] = sum;
    }
  }

  // power_to_db(ref=máximo del clip, amin=1e-10, top_db=80) y normalización a [0, 1].
  let max = 0;
  for (let i = 0; i < mel.length; i++) if (mel[i] > max) max = mel[i];
  const image = new Float32Array(N_MELS * N_FRAMES);
  if (max <= AMIN) return image; // todo -80 dB -> 0
  const refDb = 10 * Math.log10(max);
  for (let i = 0; i < mel.length; i++) {
    const db = Math.max(10 * Math.log10(Math.max(mel[i], AMIN)) - refDb, -TOP_DB);
    image[i] = Math.min(Math.max((db + TOP_DB) / TOP_DB, 0), 1);
  }
  return image;
}
