import { Asset } from "expo-asset";
import { File } from "expo-file-system";
import type { InferenceSession } from "onnxruntime-react-native";
import { NativeModules } from "react-native";

import hannWindow from "@/assets/model/hann_window.json";
import melFilters from "@/assets/model/mel_filters.json";
import type { Command } from "@/engine/commands";

import { CONFIDENCE_THRESHOLD, DEV_SERVER_URL } from "./config";
import { classify as classifyLogits, LABELS, type Label, type Prediction } from "./labels";
import { audioToImage, N_FRAMES, N_MELS, N_SAMPLES } from "./preprocess";

const REMOTE_TIMEOUT_MS = 5000;

export type VoiceResult = {
  command: Command | null; // null = no-op ("unknown" o confianza bajo el umbral)
  label: Label;
  confidence: number;
};

// Contrato común: el simulador usa MockClassifier y la app con micrófono usará OnnxClassifier.
export interface VoiceClassifier {
  classify(samples: Float32Array): Promise<VoiceResult>;
}

function toResult({ command, label, confidence }: Prediction): VoiceResult {
  return { command, label, confidence };
}

// Logits falsos cuyo softmax da `confidence` a la etiqueta elegida, para que el mock pase
// por el mismo classify() de labels.ts (umbral y no-op de "unknown") que el modelo real.
function fakeLogits(label: Label, confidence: number): number[] {
  const others = LABELS.length - 1;
  const target = confidence >= 1 ? 20 : Math.log((confidence * others) / (1 - confidence));
  return LABELS.map((l) => (l === label ? target : 0));
}

/**
 * Sin micrófono ni modelo: el simulador elige con queue() qué "se escuchó" y luego llama a
 * classify(). Las muestras de audio se ignoran.
 */
export class MockClassifier implements VoiceClassifier {
  private next: { label: Label; confidence: number } = { label: "unknown", confidence: 1 };

  queue(label: Label, confidence = 1): void {
    if (!(confidence > 1 / LABELS.length && confidence <= 1)) {
      throw new Error(`La confianza debe estar en (${(1 / LABELS.length).toFixed(3)}, 1]`);
    }
    this.next = { label, confidence };
  }

  async classify(_samples: Float32Array): Promise<VoiceResult> {
    const { label, confidence } = this.next;
    return toResult(classifyLogits(fakeLogits(label, confidence), CONFIDENCE_THRESHOLD));
  }
}

/**
 * SOLO DESARROLLO: manda las 16000 muestras a scripts/dev_server.mjs (onnxruntime-node en la PC),
 * para probar micrófono + modelo en Expo Go. El objetivo final es OnnxClassifier en Android.
 */
export class RemoteClassifier implements VoiceClassifier {
  private readonly url: string;

  constructor(baseUrl: string = DEV_SERVER_URL) {
    if (!baseUrl) throw new Error("Falta EXPO_PUBLIC_DEV_SERVER_URL en mobile/.env.local");
    this.url = `${baseUrl.replace(/\/+$/, "")}/classify`;
  }

  async classify(samples: Float32Array): Promise<VoiceResult> {
    if (samples.length !== N_SAMPLES) throw new Error(`Se requieren ${N_SAMPLES} muestras`);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REMOTE_TIMEOUT_MS);
    try {
      const response = await fetch(this.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ samples: Array.from(samples) }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Servidor: ${response.status} ${await response.text()}`);
      return (await response.json()) as VoiceResult;
    } catch (error) {
      if (controller.signal.aborted) throw new Error(`Sin respuesta de ${this.url} en ${REMOTE_TIMEOUT_MS / 1000} s`);
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }
}

type Ort = typeof import("onnxruntime-react-native");
type Model = { ort: Ort; session: InferenceSession };

let model: Promise<Model> | null = null; // una sola InferenceSession para toda la app

// onnxruntime-react-native se carga con require() al primer uso: al importarlo llama a su módulo
// nativo, que no existe en Expo Go, y un import arriba rompería la app en el iPhone.
async function createModel(): Promise<Model> {
  if (!NativeModules.Onnxruntime) {
    throw new Error("Este binario no incluye ONNX Runtime (¿Expo Go?): usa el dev build de Android");
  }
  const ort: Ort = require("onnxruntime-react-native");
  const [asset] = await Asset.loadAsync(require("@/assets/model/model.onnx"));
  if (!asset.localUri) throw new Error("No se pudo copiar model.onnx al teléfono");
  const bytes = await new File(asset.localUri).bytes();
  // Solo CPU: mismo cálculo que onnxruntime en la PC (NNAPI podría cambiar los resultados).
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["cpu"] });
  return { ort, session };
}

/**
 * Clasificador real, todo en el teléfono: 16000 muestras -> espectrograma -> model.onnx -> classify().
 * Requiere onnxruntime-react-native (dev build); no hace ninguna llamada de red.
 */
export class OnnxClassifier implements VoiceClassifier {
  async classify(samples: Float32Array): Promise<VoiceResult> {
    const image = audioToImage(samples, melFilters.data, hannWindow.data); // [64][101], valida 16000 muestras
    const logits = await this.runModel(image);
    return toResult(classifyLogits(logits, CONFIDENCE_THRESHOLD));
  }

  /** Crea la sesión si aún no existe (cargar el modelo tarda más que una inferencia). */
  async load(): Promise<void> {
    model ??= createModel().catch((error) => {
      model = null; // permite reintentar
      throw error;
    });
    await model;
  }

  /** Espectrograma [64*101] (orden [m*101+t]) -> 11 logits, en el orden de LABELS. */
  async runModel(image: Float32Array): Promise<Float32Array> {
    await this.load();
    const { ort, session } = await model!;
    const input = new ort.Tensor("float32", image, [1, 1, N_MELS, N_FRAMES]);
    const output = await session.run({ input });
    return output.logits.data as Float32Array;
  }
}
