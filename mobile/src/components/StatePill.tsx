import { StyleSheet, Text, View } from "react-native";

import { useTheme, type Theme } from "@/theme";

export type PillState = "ON" | "OFF" | "OPEN" | "GO" | "STOP" | "YES" | "BACK";

type Props = {
  state: PillState;
};

// Píldora de estado (.pill del prototipo): ON en verde, OFF en rojo y el
// resto (acciones) en violeta.
export function StatePill({ state }: Props) {
  const theme = useTheme();
  const styles = createStyles(theme);

  const colors =
    state === "ON"
      ? { backgroundColor: theme.colors.onBg, color: theme.colors.onText }
      : state === "OFF"
        ? { backgroundColor: theme.colors.offBg, color: theme.colors.offText }
        : { backgroundColor: theme.colors.violetBg, color: theme.colors.text };

  return (
    <View style={[styles.pill, { backgroundColor: colors.backgroundColor }]}>
      <Text style={[styles.text, { color: colors.color }]}>{state}</Text>
    </View>
  );
}

function createStyles(theme: Theme) {
  return StyleSheet.create({
    pill: {
      alignSelf: "flex-start",
      borderRadius: theme.radius.pill,
      paddingVertical: theme.space[1],
      paddingHorizontal: 10,
    },
    text: {
      fontSize: theme.fontSize.pill,
      fontWeight: "800",
      letterSpacing: 0.4,
    },
  });
}
