import { Asset } from "expo-asset";
import { File } from "expo-file-system";
import type { InferenceSession } from "onnxruntime-react-native";
import { NativeModules } from "react-native";

import hannWindow from "@/assets/model/hann_window.json";
import melFilters from "@/assets/model/mel_filters.json";
import type { Command } from "@/engine/commands";

import { CONFIDENCE_THRESHOLD } from "./config";
import { classify as classifyLogits, type Label, type Prediction } from "./labels";
import { audioToImage, N_FRAMES, N_MELS } from "./preprocess";

export type VoiceResult = {
  command: Command | null; // null = no-op ("unknown" o confianza bajo el umbral)
  label: Label;
  confidence: number;
  probabilities: number[]; // softmax de las 11 clases, en el orden de LABELS
};

export interface VoiceClassifier {
  classify(samples: Float32Array): Promise<VoiceResult>;
}

function toResult({ command, label, confidence, probabilities }: Prediction): VoiceResult {
  return { command, label, confidence, probabilities };
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
  /** Tiempos de la última llamada a classify(), en ms. */
  lastTiming = { preprocessMs: 0, inferenceMs: 0 };

  async classify(samples: Float32Array): Promise<VoiceResult> {
    await this.load(); // fuera de la medición: la primera vez carga el modelo
    const start = performance.now();
    const image = audioToImage(samples, melFilters.data, hannWindow.data); // [64][101], valida 16000 muestras
    const preprocessed = performance.now();
    const logits = await this.runModel(image);
    this.lastTiming = { preprocessMs: preprocessed - start, inferenceMs: performance.now() - preprocessed };
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
