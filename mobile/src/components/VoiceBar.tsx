import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { useEngineStore } from "@/engine/store";
import { useTheme, type Theme } from "@/theme";
import { captureAudio, peakOf } from "@/voice/capture";
import { OnnxClassifier, type VoiceResult } from "@/voice/classifier";
import { SILENCE_PEAK_THRESHOLD } from "@/voice/config";
import { saveCaptureWav } from "@/voice/debugWav";
import { LABELS } from "@/voice/labels";

const onnx = new OnnxClassifier();

type Status = "idle" | "recording" | "classifying";
const STATUS_TEXT: Record<Exclude<Status, "idle">, string> = {
  recording: "Grabando…",
  classifying: "Clasificando…",
};

// Solo __DEV__: top 3 y tiempos en Metro, y la ventana que entró al modelo como WAV (scripts/check_wav.mjs).
function logDebug(samples: Float32Array, { probabilities }: VoiceResult) {
  const top3 = LABELS.map((label, i) => ({ label, p: probabilities[i] }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3);
  const { preprocessMs, inferenceMs } = onnx.lastTiming;
  console.log(
    `[capture] top 3: ${top3.map(({ label, p }) => `${label}=${p.toFixed(3)}`).join(", ")} · ` +
      `espectrograma ${preprocessMs.toFixed(1)} ms · modelo ${inferenceMs.toFixed(1)} ms`,
  );
  try {
    console.log(`[capture] ventana guardada: files/captures/${saveCaptureWav(samples)}`);
  } catch (e) {
    console.warn(`[capture] no se pudo guardar el WAV: ${e instanceof Error ? e.message : String(e)}`);
  }
}

// "Hablar" graba con el micrófono y clasifica con OnnxClassifier, todo en el teléfono (dev build),
// y debajo muestra el estado o el último comando reconocido con su confianza. Si la ventana
// capturada es casi silencio (pico < SILENCE_PEAK_THRESHOLD) no se ejecuta el modelo: "Sin voz".
export function VoiceBar() {
  const dispatch = useEngineStore((s) => s.dispatch);
  const [last, setLast] = useState<VoiceResult | "silence" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const insets = useSafeAreaInsets();
  const styles = createStyles(useTheme());

  async function speak() {
    try {
      setStatus("recording");
      setError(null);
      const samples = await captureAudio();
      const peak = peakOf(samples);
      if (peak < SILENCE_PEAK_THRESHOLD) {
        // Silencio: no se ejecuta el modelo (no-op).
        if (__DEV__) console.log(`[capture] sin voz: pico ${peak.toFixed(4)} < ${SILENCE_PEAK_THRESHOLD}`);
        setLast("silence");
        return;
      }
      setStatus("classifying");
      const result = await onnx.classify(samples);
      if (__DEV__) logDebug(samples, result);
      setLast(result);
      if (result.command) dispatch(result.command); // null = no-op
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStatus("idle");
    }
  }

  const text =
    status !== "idle"
      ? STATUS_TEXT[status]
      : error !== null
        ? `Error: ${error}`
        : last === "silence"
          ? "Sin voz"
          : `Último: ${last ? `${last.label} (${last.confidence.toFixed(2)})` : "—"}`;

  return (
    <View style={[styles.container, { paddingBottom: insets.bottom + 8 }]}>
      <Pressable
        onPress={speak}
        disabled={status !== "idle"}
        style={({ pressed }) => [styles.speak, (pressed || status !== "idle") && styles.pressed]}
      >
        <Text style={styles.buttonText}>Hablar</Text>
      </Pressable>
      <Text style={[styles.label, status === "idle" && error !== null && styles.error]}>{text}</Text>
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
