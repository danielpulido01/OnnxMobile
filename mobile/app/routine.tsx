import { StyleSheet, View } from "react-native";

import { FocusCard } from "@/components/FocusCard";
import { ListeningPill } from "@/components/ListeningPill";
import { ScreenHeader } from "@/components/ScreenHeader";
import { StatePill } from "@/components/StatePill";
import { Strong } from "@/components/Strong";
import { VoiceHint } from "@/components/VoiceHint";
import { ROUTINES, type Routine } from "@/engine/handleCommand";
import { useEngineStore } from "@/engine/store";
import { useTheme, type Theme } from "@/theme";

const ROUTINE_TEXT: Record<Routine, { title: string; description: string }> = {
  Study: {
    title: "Study Mode",
    description: "Desk light on, fan low, notifications muted.",
  },
  Night: {
    title: "Night Mode",
    description: "Warm lights, TV off, reduced brightness.",
  },
  "Welcome Home": {
    title: "Welcome Home",
    description: "Lights on and ambient music prepared.",
  },
};

// Pantalla 3/4. Solo lee el store; los comandos se procesan en src/engine.
export default function RoutineScreen() {
  const focus = useEngineStore((s) => s.routineFocus);
  const running = useEngineStore((s) => s.runningRoutine);
  const styles = createStyles(useTheme());

  return (
    <View style={styles.screen}>
      <ListeningPill label="go / stop / up / down / no" />
      <ScreenHeader title="Routines" screen={3} />

      <View style={styles.stack}>
        {ROUTINES.map((routine, index) => (
          <FocusCard
            key={routine}
            row
            title={ROUTINE_TEXT[routine].title}
            description={ROUTINE_TEXT[routine].description}
            focused={index === focus}
            // La rutina en ejecución muestra STOP (se puede interrumpir); las demás, GO.
            pill={<StatePill state={index === running ? "STOP" : "GO"} />}
          />
        ))}
      </View>

      <View style={styles.spacer} />

      <VoiceHint>
        move with <Strong>up/down</Strong>, activate with <Strong>go</Strong>,
        interrupt with <Strong>stop</Strong>; <Strong>no</Strong> returns.
      </VoiceHint>
    </View>
  );
}

function createStyles(theme: Theme) {
  return StyleSheet.create({
    screen: {
      flex: 1,
      padding: theme.space[4],
      gap: theme.space[4],
    },
    stack: {
      gap: theme.space[3],
    },
    spacer: {
      flex: 1,
    },
  });
}
