import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { COMMANDS } from "@/engine/commands";
import { useEngineStore } from "@/engine/store";
import { useTheme, type Theme } from "@/theme";
import { captureAudio } from "@/voice/capture";
import { MockClassifier, OnnxClassifier, type VoiceResult } from "@/voice/classifier";
import type { Label } from "@/voice/labels";
import { runModelTest, type ModelTestReport } from "@/voice/modelTest";
import { N_SAMPLES } from "@/voice/preprocess";

const SIMULATED: Label[] = [...COMMANDS, "unknown"];
const classifier = new MockClassifier();
const SILENCE = new Float32Array(N_SAMPLES); // el mock ignora el audio
const onnx = new OnnxClassifier();
const MAX_LOGIT_DIFF = 1e-4; // en la PC (Node) la diferencia fue 7.6e-6

type Status = "idle" | "recording" | "sending" | "testing";
const SPEAK_TEXT: Record<Status, string> = {
  idle: "🎤 Hablar",
  recording: "Grabando…",
  sending: "Clasificando…",
  testing: "🎤 Hablar",
};

function formatTest({ loadMs, items }: ModelTestReport): string {
  const lines = items.map(
    (item) =>
      `${item.label} (${item.confidence.toFixed(3)}) ${item.ms.toFixed(1)} ms · Δlogits ${item.maxDiff.toExponential(1)} ` +
      (item.label === item.expectedLabel && item.maxDiff < MAX_LOGIT_DIFF ? "✓" : `✗ (PC: ${item.expectedLabel})`),
  );
  return [`Modelo cargado en ${loadMs.toFixed(0)} ms`, ...lines].join("\n");
}

// Botones para probar el flujo sin micrófono ni modelo: cada botón simula lo que
// "se escuchó" y pasa por el mismo VoiceClassifier -> dispatch que usará el modelo ONNX.
// "Hablar" graba con el micrófono y clasifica con OnnxClassifier, todo en el teléfono (dev build).
export function CommandSimulator() {
  const dispatch = useEngineStore((s) => s.dispatch);
  const [last, setLast] = useState<VoiceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [info, setInfo] = useState<string | null>(null);
  const insets = useSafeAreaInsets();
  const styles = createStyles(useTheme());

  function apply(result: VoiceResult) {
    setLast(result);
    setError(null);
    if (result.command) dispatch(result.command); // null = no-op
  }

  async function simulate(label: Label) {
    classifier.queue(label);
    apply(await classifier.classify(SILENCE));
  }

  async function speak() {
    try {
      setStatus("recording");
      setInfo(null);
      const samples = await captureAudio();
      setStatus("sending");
      const result = await onnx.classify(samples);
      const { preprocessMs, inferenceMs } = onnx.lastTiming;
      let peak = 0;
      for (const value of samples) peak = Math.max(peak, Math.abs(value));
      setInfo(
        `🎤 ${result.label} (${result.confidence.toFixed(3)}) · modelo ${inferenceMs.toFixed(1)} ms · ` +
          `espectrograma ${preprocessMs.toFixed(1)} ms · pico ${peak.toPrecision(2)}`,
      );
      apply(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStatus("idle");
    }
  }

  async function testModel() {
    try {
      setStatus("testing");
      setInfo(null);
      setInfo(formatTest(await runModelTest(onnx)));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStatus("idle");
    }
  }

  return (
    <View style={[styles.container, { paddingBottom: insets.bottom + 8 }]}>
      <Text style={[styles.label, error !== null && styles.error]}>
        {error !== null
          ? `Error: ${error}`
          : `Simulador · último: ${
              last ? `${last.label} (${last.confidence.toFixed(2)}) → ${last.command ?? "no-op"}` : "—"
            }`}
      </Text>
      <Pressable
        onPress={speak}
        disabled={status !== "idle"}
        style={({ pressed }) => [styles.speak, (pressed || status !== "idle") && styles.pressed]}
      >
        <Text style={styles.buttonText}>{SPEAK_TEXT[status]}</Text>
      </Pressable>
      <Pressable
        onPress={testModel}
        disabled={status !== "idle"}
        style={({ pressed }) => [styles.button, styles.wide, (pressed || status !== "idle") && styles.pressed]}
      >
        <Text style={styles.buttonText}>{status === "testing" ? "Probando…" : "Probar modelo"}</Text>
      </Pressable>
      {info !== null && <Text style={styles.label}>{info}</Text>}
      <View style={styles.grid}>
        {SIMULATED.map((label) => (
          <Pressable
            key={label}
            onPress={() => simulate(label)}
            style={({ pressed }) => [styles.button, pressed && styles.pressed]}
          >
            <Text style={styles.buttonText}>{label}</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

function createStyles(theme: Theme) {
  return StyleSheet.create({
    container: {
      paddingTop: theme.space[2],
      paddingHorizontal: theme.space[4],
      gap: theme.space[2],
      backgroundColor: theme.colors.surface,
      borderTopWidth: 1,
      borderTopColor: theme.colors.border,
    },
    label: {
      fontSize: theme.fontSize.small,
      color: theme.colors.text2,
      textAlign: "center",
    },
    grid: {
      flexDirection: "row",
      flexWrap: "wrap",
      justifyContent: "center",
      gap: theme.space[2],
    },
    button: {
      width: "18%",
      paddingVertical: 10,
      borderRadius: theme.radius.pill,
      backgroundColor: theme.colors.violetBg,
      alignItems: "center",
    },
    wide: {
      width: "100%",
    },
    error: {
      color: theme.colors.offText,
    },
    speak: {
      paddingVertical: 12,
      borderRadius: theme.radius.pill,
      backgroundColor: theme.colors.accentBg,
      borderWidth: 1,
      borderColor: theme.colors.accent,
      alignItems: "center",
    },
    pressed: {
      opacity: 0.6,
    },
    buttonText: {
      color: theme.colors.text,
      fontWeight: "600",
    },
  });
}
