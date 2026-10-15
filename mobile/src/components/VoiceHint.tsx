import type { ReactNode } from "react";
import { StyleSheet, Text, View } from "react-native";

import { useTheme, type Theme } from "@/theme";
import { Strong } from "./Strong";

type Props = {
  // Explicación del flujo de voz; los comandos se marcan con <Strong>.
  children: ReactNode;
};

// Caja "Voice flow: ..." al pie de cada pantalla (.hint del prototipo).
export function VoiceHint({ children }: Props) {
  const styles = createStyles(useTheme());

  return (
    <View style={styles.hint}>
      <Text style={styles.text}>
        <Strong>Voice flow:</Strong> {children}
      </Text>
    </View>
  );
}

function createStyles(theme: Theme) {
  return StyleSheet.create({
    hint: {
      backgroundColor: theme.colors.surface,
      borderWidth: 1,
      borderColor: theme.colors.border,
      borderRadius: theme.radius.card,
      padding: theme.space[3],
    },
    text: {
      color: theme.colors.text2,
      fontSize: theme.fontSize.small,
      lineHeight: 15,
    },
  });
}
