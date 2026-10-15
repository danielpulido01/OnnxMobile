import { useColorScheme } from "react-native";

// Tokens de diseño tomados de docs/prototipo.html.
// Los colores cambian entre modo oscuro y claro; forma, espaciado y letra
// son iguales en ambos. Los nombres siguen las variables CSS del prototipo
// (--card-focus -> cardFocus, --space-4 -> space[4], --fs-title -> fontSize.title).

const shared = {
  radius: {
    card: 16,
    pill: 999,
  },
  space: {
    1: 4,
    2: 8,
    3: 12,
    4: 16,
    5: 20,
    6: 24,
  },
  fontSize: {
    title: 20,
    cardTitle: 14,
    body: 12,
    small: 11,
    pill: 10, // .pill del prototipo (no es variable de :root, pero es fijo)
  },
};

// :root (modo oscuro, el principal del prototipo)
export const darkTheme = {
  ...shared,
  colors: {
    bg: "#0a0f1f",
    surface: "#131a2e",
    card: "#171e33",
    cardFocus: "#1b2a3d",
    border: "#252d47",
    text: "#f2f5ff",
    text2: "#9aa5c6",
    accent: "#4fe3c1",
    accentBg: "#12332f",
    violet: "#8b93ff",
    violetBg: "#232a4d",
    onBg: "#12332f",
    onText: "#6ee7c7",
    offBg: "#3a1c28",
    offText: "#f58fa3",
    dangerCard: "#2a1a26",
  },
};

export type Theme = typeof darkTheme;

// :root[data-theme="light"] (modo claro)
export const lightTheme: Theme = {
  ...shared,
  colors: {
    bg: "#eef1fa",
    surface: "#ffffff",
    card: "#ffffff",
    cardFocus: "#e6f8f3",
    border: "#d5dbee",
    text: "#12172b",
    text2: "#5a6486",
    accent: "#0f9d84",
    accentBg: "#dcf5ee",
    violet: "#5b63e6",
    violetBg: "#e3e5fb",
    onBg: "#d9f4ea",
    onText: "#0b7a64",
    offBg: "#fde3e8",
    offText: "#c0304f",
    dangerCard: "#fdeef1",
  },
};

// Sigue el tema del sistema. Si el sistema no informa ninguno, usa el oscuro,
// que es el modo principal del prototipo.
export function useTheme(): Theme {
  const scheme = useColorScheme();
  return scheme === "light" ? lightTheme : darkTheme;
}
