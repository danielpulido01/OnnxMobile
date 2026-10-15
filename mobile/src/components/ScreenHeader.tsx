import { StyleSheet, Text, View } from "react-native";

import { useTheme, type Theme } from "@/theme";

const TOTAL_SCREENS = 4;

type Props = {
  title: string;
  // Número de la pantalla (1 a 4) para el texto "Screen n/4".
  screen: number;
};

// Título de la pantalla + "Screen n/4" (.head del prototipo).
export function ScreenHeader({ title, screen }: Props) {
  const styles = createStyles(useTheme());

  return (
    <View style={styles.head}>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.counter}>
        Screen {screen}/{TOTAL_SCREENS}
      </Text>
    </View>
  );
}

function createStyles(theme: Theme) {
  return StyleSheet.create({
    head: {
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "baseline",
      gap: theme.space[3],
    },
    title: {
      flexShrink: 1,
      color: theme.colors.text,
      fontSize: theme.fontSize.title,
      fontWeight: "bold",
    },
    counter: {
      color: theme.colors.text2,
      fontSize: theme.fontSize.small,
    },
  });
}
