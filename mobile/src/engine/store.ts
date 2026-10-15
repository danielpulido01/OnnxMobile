import { router } from "expo-router";
import { create } from "zustand";

import type { Command } from "./commands";
import { handleCommand, type EngineState, type Navigation } from "./handleCommand";

// Resultado del último comando, para dar feedback al usuario.
// id aumenta en cada comando, así repetir el mismo comando vuelve a mostrar el aviso.
export type CommandResult = {
  id: number;
  command: Command;
  accepted: boolean;
};

type EngineStore = EngineState & {
  screen: string;
  lastResult: CommandResult | null;
  setScreen: (screen: string) => void;
  // Punto de entrada único: lo usa el simulador ahora y el modelo ONNX después.
  dispatch: (command: Command) => void;
};

export const useEngineStore = create<EngineStore>()((set, get) => ({
  dashboardFocus: 0,
  selectedDevice: "lights",
  devices: { lights: false, fan: false, tv: false },
  brightnessMode: false,
  routineFocus: 0,
  runningRoutine: null,

  screen: "/",
  lastResult: null,

  setScreen: (screen) => set({ screen }),

  dispatch: (command) => {
    const state = get();
    const result = handleCommand(state.screen, state, command);
    set({
      ...result?.patch,
      lastResult: {
        id: (state.lastResult?.id ?? 0) + 1,
        command,
        accepted: result !== null,
      },
    });
    if (result?.nav) navigate(result.nav);
  },
}));

function navigate(nav: Navigation) {
  if (nav.type === "push") {
    router.push(nav.to);
  } else if (router.canGoBack()) {
    router.back();
  }
}
