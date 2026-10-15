import type { ReactNode } from "react";
import { Text } from "react-native";

import { useTheme } from "@/theme";

// Texto en negrita dentro de otro <Text> (equivale a <b> del prototipo).
export function Strong({ children }: { children: ReactNode }) {
  const theme = useTheme();
  return <Text style={{ color: theme.colors.text, fontWeight: "bold" }}>{children}</Text>;
}
