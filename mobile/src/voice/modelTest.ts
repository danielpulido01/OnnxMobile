import { Asset } from "expo-asset";
import { File } from "expo-file-system";

import hannWindow from "@/assets/model/hann_window.json";
import melFilters from "@/assets/model/mel_filters.json";
import reference from "@/assets/test/reference_logits.json";

import { parseWav } from "./capture";
import type { OnnxClassifier } from "./classifier";
import { CONFIDENCE_THRESHOLD } from "./config";
import { classify, type Label } from "./labels";
import { audioToImage, fixLength } from "./preprocess";

// Mismo nombre que en assets/test/ (require() necesita rutas fijas).
const WAVS: Record<string, number> = {
  yes__cc8b3228_nohash_0: require("@/assets/test/yes__cc8b3228_nohash_0.wav"),
  go__471a0925_nohash_2: require("@/assets/test/go__471a0925_nohash_2.wav"),
};

export type ModelTestItem = {
  name: string;
  label: Label;
  confidence: number;
  expectedLabel: string;
  maxDiff: number; // máx |logit teléfono - logit PC|
  ms: number; // solo session.run(), sin cargar el modelo
};

export type ModelTestReport = { loadMs: number; items: ModelTestItem[] };

/**
 * WAV de referencia -> parseWav -> fixLength -> audioToImage -> model.onnx, todo en el teléfono,
 * y compara los logits con los de la PC (assets/test/reference_logits.json).
 */
export async function runModelTest(classifier: OnnxClassifier): Promise<ModelTestReport> {
  const loadStart = performance.now();
  await classifier.load();
  const loadMs = performance.now() - loadStart;

  const items: ModelTestItem[] = [];
  for (const expected of reference.items) {
    const [asset] = await Asset.loadAsync(WAVS[expected.name]);
    const { samples } = parseWav(await new File(asset.localUri!).bytes());
    const image = audioToImage(fixLength(samples), melFilters.data, hannWindow.data);

    const start = performance.now();
    const logits = await classifier.runModel(image);
    const ms = performance.now() - start;

    let maxDiff = 0;
    expected.logits.forEach((value, i) => (maxDiff = Math.max(maxDiff, Math.abs(logits[i] - value))));
    const { label, confidence } = classify(logits, CONFIDENCE_THRESHOLD);
    console.log(`[modelTest] ${expected.name}: logits=${JSON.stringify(Array.from(logits))} maxDiff=${maxDiff.toExponential(3)}`);
    items.push({ name: expected.name, label, confidence, expectedLabel: expected.predicted_label, maxDiff, ms });
  }
  return { loadMs, items };
}
