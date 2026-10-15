import { StyleSheet, View } from "react-native";

import { FocusCard } from "@/components/FocusCard";
import { ListeningPill } from "@/components/ListeningPill";
import { ScreenHeader } from "@/components/ScreenHeader";
import { Strong } from "@/components/Strong";
import { VoiceHint } from "@/components/VoiceHint";
import { useTheme, type Theme } from "@/theme";

// Pantalla 4/4. No tiene estado propio; yes/no se procesan en src/engine.
export default function Confirm() {
  const styles = createStyles(useTheme());

  return (
    <View style={styles.screen}>
      <ListeningPill label="yes / no" />
      <ScreenHeader title="Confirm action" screen={4} />

      <FocusCard
        title="Turn off all lights?"
        description={
          <>
            This confirmation view appears after a sensitive command. The user
            must explicitly say <Strong>yes</Strong> to accept or{" "}
            <Strong>no</Strong> to cancel and return.
          </>
        }
      >
        <View style={styles.grid}>
          <FocusCard title="yes" description="Execute the action." focused style={styles.option} />
          <FocusCard title="no" description="Cancel and go back." style={styles.option} />
        </View>
      </FocusCard>

      {/* Empuja VoiceHint al pie de la pantalla, como en el prototipo. */}
      <View style={styles.spacer} />

      <VoiceHint>
        confirmation screens reduce ambiguity and make the application safer and
        more structured.
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
    grid: {
      flexDirection: "row",
      gap: theme.space[3],
      marginTop: theme.space[4],
    },
    option: {
      flex: 1,
    },
    spacer: {
      flex: 1,
    },
  });
}
