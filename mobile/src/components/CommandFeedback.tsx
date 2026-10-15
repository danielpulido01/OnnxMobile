import { useEffect, useState } from "react";
import { StyleSheet, Text } from "react-native";
import Animated, { FadeInUp, FadeOut } from "react-native-reanimated";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { useEngineStore } from "@/engine/store";
import { useTheme, type Theme } from "@/theme";

const VISIBLE_MS = 1500;

// Aviso breve arriba de la pantalla después de cada comando:
// verde si se aplicó, rojo si no aplica en la pantalla actual.
export function CommandFeedback() {
  const result = useEngineStore((s) => s.lastResult);
  // id del último aviso que ya se ocultó.
  const [hiddenId, setHiddenId] = useState<number | null>(null);
  const insets = useSafeAreaInsets();
  const styles = createStyles(useTheme());

  useEffect(() => {
    if (!result) return;
    const timer = setTimeout(() => setHiddenId(result.id), VISIBLE_MS);
    // Si llega otro comando antes, se cancela este timer y empieza uno nuevo.
    return () => clearTimeout(timer);
  }, [result]);

  if (!result || result.id === hiddenId) return null;

  return (
    <Animated.View
      // key nueva por comando: la animación de entrada se repite cada vez.
      key={result.id}
      entering={FadeInUp.duration(150)}
      exiting={FadeOut.duration(150)}
      pointerEvents="none"
      style={[
        styles.toast,
        result.accepted ? styles.accepted : styles.rejected,
        { top: insets.top + 8 },
      ]}
    >
      <Text style={[styles.text, result.accepted ? styles.acceptedText : styles.rejectedText]}>
        {result.accepted
          ? `✓ Heard: ${result.command}`
          : `“${result.command}” isn't available here`}
      </Text>
    </Animated.View>
  );
}

function createStyles(theme: Theme) {
  return StyleSheet.create({
    toast: {
      position: "absolute",
      alignSelf: "center",
      paddingVertical: theme.space[2],
      paddingHorizontal: theme.space[4],
      borderRadius: theme.radius.pill,
    },
    accepted: {
      backgroundColor: theme.colors.onBg,
    },
    rejected: {
      backgroundColor: theme.colors.offBg,
    },
    text: {
      fontSize: theme.fontSize.body,
      fontWeight: "700",
    },
    acceptedText: {
      color: theme.colors.onText,
    },
    rejectedText: {
      color: theme.colors.offText,
    },
  });
}
