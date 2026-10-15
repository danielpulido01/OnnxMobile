import { StyleSheet, View } from "react-native";

import { FocusCard } from "@/components/FocusCard";
import { ListeningPill } from "@/components/ListeningPill";
import { ScreenHeader } from "@/components/ScreenHeader";
import { StatePill } from "@/components/StatePill";
import { Strong } from "@/components/Strong";
import { VoiceHint } from "@/components/VoiceHint";
import type { DeviceId } from "@/engine/handleCommand";
import { useEngineStore } from "@/engine/store";
import { useTheme, type Theme } from "@/theme";

const DEVICE_TITLES: Record<DeviceId, string> = {
  lights: "Living Room Lights",
  fan: "Fan",
  tv: "TV",
};

// Pantalla 2/4. Solo lee el store; los comandos se procesan en src/engine.
export default function Device() {
  const device = useEngineStore((s) => s.selectedDevice);
  const isOn = useEngineStore((s) => s.devices[s.selectedDevice]);
  const brightnessMode = useEngineStore((s) => s.brightnessMode);
  const styles = createStyles(useTheme());
  // Brightness mode (y el comando yes) solo aplica a las luces.
  const hasBrightness = device === "lights";

  return (
    <View style={styles.screen}>
      <ListeningPill label={hasBrightness ? "on / off / yes / no" : "on / off / no"} />
      <ScreenHeader title={DEVICE_TITLES[device]} screen={2} />

      <View style={styles.stack}>
        <FocusCard
          row
          title="Current state"
          description="Primary device selected"
          pill={<StatePill state={isOn ? "ON" : "OFF"} />}
        />

        {/* El foco marca la acción que corresponde al estado actual. */}
        <View style={styles.grid}>
          <FocusCard
            title="Turn On"
            description={<>Activate with command <Strong>on</Strong>.</>}
            focused={isOn}
            style={styles.action}
          />
          <FocusCard
            title="Turn Off"
            description={<>Deactivate with command <Strong>off</Strong>.</>}
            focused={!isOn}
            danger
            style={styles.action}
          />
        </View>

        {hasBrightness && (
          <FocusCard
            row
            title="Brightness mode"
            description={
              !isOn ? (
                <>Turn the lights <Strong>on</Strong> first.</>
              ) : brightnessMode ? (
                <>Active. Say <Strong>yes</Strong> to turn it off.</>
              ) : (
                <>Say <Strong>yes</Strong> to adjust scene intensity.</>
              )
            }
            focused={brightnessMode}
            pill={<StatePill state="YES" />}
          />
        )}
        <FocusCard
          row
          title="Back to dashboard"
          description={<>Return using the command <Strong>no</Strong>.</>}
          pill={<StatePill state="BACK" />}
        />
      </View>

      <VoiceHint>
        <Strong>on/off</Strong> control the selected device;{" "}
        {hasBrightness && (
          <>
            <Strong>yes</Strong> confirms a highlighted action;{" "}
          </>
        )}
        <Strong>no</Strong> returns.
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
      flex: 1,
      gap: theme.space[3],
    },
    grid: {
      flexDirection: "row",
      gap: theme.space[3],
    },
    action: {
      flex: 1,
      paddingVertical: theme.space[5],
    },
  });
}
