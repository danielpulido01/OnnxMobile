import { Pressable, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { COMMANDS } from "@/engine/commands";
import { useEngineStore } from "@/engine/store";
import { useTheme, type Theme } from "@/theme";

// Botones para probar el motor sin el modelo: cada botón envía un comando
// al mismo dispatch que usará la salida del modelo ONNX.
export function CommandSimulator() {
  const dispatch = useEngineStore((s) => s.dispatch);
  const lastCommand = useEngineStore((s) => s.lastResult?.command);
  const insets = useSafeAreaInsets();
  const styles = createStyles(useTheme());

  return (
    <View style={[styles.container, { paddingBottom: insets.bottom + 8 }]}>
      <Text style={styles.label}>
        Simulador · último comando: {lastCommand ?? "—"}
      </Text>
      <View style={styles.grid}>
        {COMMANDS.map((command) => (
          <Pressable
            key={command}
            onPress={() => dispatch(command)}
            style={({ pressed }) => [styles.button, pressed && styles.pressed]}
          >
            <Text style={styles.buttonText}>{command}</Text>
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
    pressed: {
      opacity: 0.6,
    },
    buttonText: {
      color: theme.colors.text,
      fontWeight: "600",
    },
  });
}
