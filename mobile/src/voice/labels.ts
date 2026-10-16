import type { Command } from "../engine/commands";

// Orden de los 11 logits del modelo: "classes" de assets/model/preprocessing.json.
// Ojo: no es el mismo orden que COMMANDS en src/engine/commands.ts.
export const LABELS = [
  "yes",
  "no",
  "left",
  "right",
  "up",
  "down",
  "on",
  "off",
  "stop",
  "go",
  "unknown",
] as const satisfies readonly (Command | "unknown")[];

export type Label = (typeof LABELS)[number];

export type Prediction = {
  index: number;
  label: Label;
  confidence: number; // probabilidad softmax de la clase ganadora
  probabilities: number[];
  // Comando a ejecutar, o null si es "unknown" o la confianza no llega al umbral (no-op).
  command: Command | null;
};

// Softmax numéricamente estable (resta el máximo antes de exp).
export function softmax(logits: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const exps = Array.from(logits, (value) => Math.exp(value - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((value) => value / sum);
}

export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/**
 * Interpreta los logits del modelo.
 * @param threshold confianza mínima en [0, 1] para aceptar un comando.
 */
export function classify(logits: ArrayLike<number>, threshold: number): Prediction {
  if (logits.length !== LABELS.length) throw new Error(`Se esperan ${LABELS.length} logits`);
  if (!(threshold >= 0 && threshold <= 1)) throw new Error("El umbral debe estar en [0, 1]");
  const probabilities = softmax(logits);
  const index = argmax(probabilities);
  const label = LABELS[index];
  const confidence = probabilities[index];
  const command = label !== "unknown" && confidence >= threshold ? label : null;
  return { index, label, confidence, probabilities, command };
}
