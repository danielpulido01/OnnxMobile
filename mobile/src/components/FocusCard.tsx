import { useEffect, type ReactNode } from "react";
import { StyleSheet, Text, View, type StyleProp, type ViewStyle } from "react-native";
import Animated, {
  interpolateColor,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from "react-native-reanimated";

import { useTheme, type Theme } from "@/theme";

const FOCUS_DURATION_MS = 250;
// Cuánto crece la tarjeta con foco (3 %), para que el cambio se note.
const FOCUS_SCALE = 0.03;

type Props = {
  title: string;
  description?: ReactNode;
  // Píldora de estado (normalmente un <StatePill>).
  pill?: ReactNode;
  // true: título a la izquierda y píldora a la derecha (.card.row).
  // false: píldora debajo del texto (tarjetas de la cuadrícula).
  row?: boolean;
  // Borde accent y fondo cardFocus cuando el foco de voz está aquí.
  focused?: boolean;
  // Fondo rojizo para acciones destructivas (.card.danger, "Turn Off").
  danger?: boolean;
  style?: StyleProp<ViewStyle>;
  // Contenido extra debajo de la descripción (p. ej. la cuadrícula de Confirm).
  children?: ReactNode;
};

// Tarjeta base del prototipo (.card / .card.focus).
export function FocusCard({
  title,
  description,
  pill,
  row = false,
  focused = false,
  danger = false,
  style,
  children,
}: Props) {
  const theme = useTheme();
  const styles = createStyles(theme);

  // progress va de 0 (sin foco) a 1 (con foco). Cuando cambia `focused`,
  // withTiming lo lleva al nuevo valor en FOCUS_DURATION_MS y Reanimated
  // recalcula el estilo en cada cuadro de la animación.
  const progress = useSharedValue(focused ? 1 : 0);

  useEffect(() => {
    progress.set(withTiming(focused ? 1 : 0, { duration: FOCUS_DURATION_MS }));
  }, [focused, progress]);

  const baseBg = danger ? theme.colors.dangerCard : theme.colors.card;
  const { border, accent, cardFocus } = theme.colors;

  const focusStyle = useAnimatedStyle(() => {
    const p = progress.get();
    return {
      borderWidth: 1 + p,
      borderColor: interpolateColor(p, [0, 1], [border, accent]),
      backgroundColor: interpolateColor(p, [0, 1], [baseBg, cardFocus]),
      transform: [{ scale: 1 + FOCUS_SCALE * p }],
    };
  });

  return (
    <Animated.View style={[styles.card, row && styles.row, style, focusStyle]}>
      <View style={row && styles.rowText}>
        <Text style={styles.title}>{title}</Text>
        {description != null && <Text style={styles.description}>{description}</Text>}
      </View>
      {pill != null && <View style={!row && styles.pillBelow}>{pill}</View>}
      {children}
    </Animated.View>
  );
}

// Borde y fondo no están aquí: los controla focusStyle (animado).
function createStyles(theme: Theme) {
  return StyleSheet.create({
    card: {
      borderRadius: theme.radius.card,
      paddingVertical: theme.space[3],
      paddingHorizontal: theme.space[4],
    },
    row: {
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "center",
      gap: theme.space[3],
    },
    rowText: {
      flexShrink: 1,
    },
    title: {
      color: theme.colors.text,
      fontSize: theme.fontSize.cardTitle,
      fontWeight: "bold",
      marginBottom: theme.space[1],
    },
    description: {
      color: theme.colors.text2,
      fontSize: theme.fontSize.small,
      lineHeight: 15,
    },
    pillBelow: {
      marginTop: theme.space[2],
    },
  });
}
