import { Stack, usePathname } from "expo-router";
import { useEffect } from "react";
import { View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { CommandFeedback } from "@/components/CommandFeedback";
import { VoiceBar } from "@/components/VoiceBar";
import { useEngineStore } from "@/engine/store";
import { useTheme } from "@/theme";

export default function RootLayout() {
  // El motor necesita saber en qué pantalla estamos. Se sincroniza aquí, en un
  // solo lugar, para que también funcione al volver con el gesto de iOS.
  const pathname = usePathname();
  const setScreen = useEngineStore((s) => s.setScreen);

  useEffect(() => {
    setScreen(pathname);
  }, [pathname, setScreen]);

  // Fondo del tema y margen superior para no quedar bajo el notch
  // (las pantallas no tienen header).
  const theme = useTheme();
  const insets = useSafeAreaInsets();

  return (
    <View style={{ flex: 1, backgroundColor: theme.colors.bg, paddingTop: insets.top }}>
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: theme.colors.bg },
        }}
      />
      <VoiceBar />
      <CommandFeedback />
    </View>
  );
}
