import { StyleSheet, View } from "react-native";

import { FocusCard } from "@/components/FocusCard";
import { ListeningPill } from "@/components/ListeningPill";
import { ScreenHeader } from "@/components/ScreenHeader";
import { StatePill } from "@/components/StatePill";
import { Strong } from "@/components/Strong";
import { VoiceHint } from "@/components/VoiceHint";
import { DASHBOARD_CARDS, type DashboardCard } from "@/engine/handleCommand";
import { useEngineStore } from "@/engine/store";
import { useTheme, type Theme } from "@/theme";

const CARDS: Record<DashboardCard, { title: string; description: string }> = {
  lights: { title: "Lights", description: "Living room and bedroom lights" },
  fan: { title: "Fan", description: "Air circulation and comfort mode" },
  tv: { title: "TV", description: "Media and ambient screen" },
  routines: { title: "Routines", description: "Study, sleep and welcome scenes" },
};

// Pantalla 1/4. Solo lee el store; los comandos se procesan en src/engine.
export default function Dashboard() {
  const focus = useEngineStore((s) => s.dashboardFocus);
  const devices = useEngineStore((s) => s.devices);
  const styles = createStyles(useTheme());

  return (
    <View style={styles.screen}>
      <ListeningPill label="navigation commands" />
      <ScreenHeader title="Home Dashboard" screen={1} />

      <View style={styles.grid}>
        {DASHBOARD_CARDS.map((card, index) => (
          <FocusCard
            key={card}
            title={CARDS[card].title}
            description={CARDS[card].description}
            focused={index === focus}
            pill={
              <StatePill
                state={card === "routines" ? "OPEN" : devices[card] ? "ON" : "OFF"}
              />
            }
            style={styles.gridItem}
          />
        ))}
      </View>

      <View style={styles.spacer} />

      <VoiceHint>
        use <Strong>up/down/left/right</Strong> to move the focus and{" "}
        <Strong>yes</Strong> to open the selected card. Say <Strong>off</Strong>{" "}
        to turn off all lights.
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
      flexWrap: "wrap",
      gap: theme.space[3],
    },
    gridItem: {
      flexBasis: "45%",
      flexGrow: 1,
    },
    spacer: {
      flex: 1,
    },
  });
}
