import { StyleSheet, Text, View } from "react-native";

import { useTheme, type Theme } from "@/theme";

type Props = {
  // Qué comandos escucha la pantalla, p. ej. "on / off / yes / no".
  label: string;
};

// Píldora "● Listening: ..." (.listening del prototipo).
export function ListeningPill({ label }: Props) {
  const styles = createStyles(useTheme());

  return (
    <View style={styles.pill}>
      <Text style={styles.text}>
        <Text style={styles.dot}>● </Text>
        Listening: {label}
      </Text>
    </View>
  );
}

function createStyles(theme: Theme) {
  return StyleSheet.create({
    pill: {
      alignSelf: "flex-start",
      backgroundColor: theme.colors.accentBg,
      borderWidth: 1,
      borderColor: theme.colors.accent,
      borderRadius: theme.radius.pill,
      paddingVertical: 6,
      paddingHorizontal: theme.space[3],
    },
    text: {
      color: theme.colors.text,
      fontSize: theme.fontSize.small,
    },
    dot: {
      color: theme.colors.accent,
    },
  });
}
