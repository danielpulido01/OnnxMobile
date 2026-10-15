import type { Command } from "./commands";

// Reglas de la app: (pantalla actual, estado, comando) -> cambios + navegación.
// Es una función pura: no usa React, zustand ni el router, así cada regla
// se puede leer y probar por separado.
//
// Devuelve null si el comando no aplica en la pantalla actual (p. ej. "go" en
// el Dashboard). Devuelve {} si el comando es válido pero no cambia nada
// (p. ej. "up" cuando el foco ya está en la fila de arriba).

export const DASHBOARD_CARDS = ["lights", "fan", "tv", "routines"] as const;
export type DashboardCard = (typeof DASHBOARD_CARDS)[number];
export type DeviceId = Exclude<DashboardCard, "routines">;

export const ROUTINES = ["Study", "Night", "Welcome Home"] as const;
export type Routine = (typeof ROUTINES)[number];

// Las tarjetas del Dashboard forman una cuadrícula de 2 columnas:
//   0 Lights | 1 Fan
//   2 TV     | 3 Routines
const GRID_COLUMNS = 2;

export type EngineState = {
  dashboardFocus: number;
  selectedDevice: DeviceId;
  devices: Record<DeviceId, boolean>;
  brightnessMode: boolean;
  routineFocus: number;
  runningRoutine: number | null;
};

export type Navigation =
  | { type: "push"; to: "/device" | "/routine" | "/confirm" }
  | { type: "back" };

export type EngineResult = {
  patch?: Partial<EngineState>;
  nav?: Navigation;
};

export function handleCommand(
  screen: string,
  state: EngineState,
  command: Command,
): EngineResult | null {
  switch (screen) {
    case "/":
      return handleDashboard(state, command);
    case "/device":
      return handleDevice(state, command);
    case "/routine":
      return handleRoutine(state, command);
    case "/confirm":
      return handleConfirm(state, command);
    default:
      return null;
  }
}

function handleDashboard(state: EngineState, command: Command): EngineResult | null {
  const focus = state.dashboardFocus;
  const row = Math.floor(focus / GRID_COLUMNS);
  const col = focus % GRID_COLUMNS;
  const lastRow = Math.floor((DASHBOARD_CARDS.length - 1) / GRID_COLUMNS);

  switch (command) {
    case "up":
      return row > 0 ? { patch: { dashboardFocus: focus - GRID_COLUMNS } } : {};
    case "down":
      return row < lastRow ? { patch: { dashboardFocus: focus + GRID_COLUMNS } } : {};
    case "left":
      return col > 0 ? { patch: { dashboardFocus: focus - 1 } } : {};
    case "right":
      return col < GRID_COLUMNS - 1 ? { patch: { dashboardFocus: focus + 1 } } : {};
    case "yes": {
      const card = DASHBOARD_CARDS[focus];
      if (card === "routines") {
        return { nav: { type: "push", to: "/routine" } };
      }
      return {
        patch: { selectedDevice: card },
        nav: { type: "push", to: "/device" },
      };
    }
    // Apagar todas las luces es una acción sensible: se pide confirmación.
    case "off":
      return { nav: { type: "push", to: "/confirm" } };
    default:
      return null;
  }
}

function handleDevice(state: EngineState, command: Command): EngineResult | null {
  const device = state.selectedDevice;

  switch (command) {
    case "on":
      return { patch: { devices: { ...state.devices, [device]: true } } };
    case "off":
      return {
        patch: {
          devices: { ...state.devices, [device]: false },
          // Apagar las luces también apaga su Brightness mode.
          ...(device === "lights" && { brightnessMode: false }),
        },
      };
    // Brightness mode solo existe para las luces y solo si están encendidas.
    // "yes" lo alterna: es la única forma de apagarlo con los 10 comandos.
    case "yes":
      if (device !== "lights" || !state.devices.lights) return null;
      return { patch: { brightnessMode: !state.brightnessMode } };
    case "no":
      return { nav: { type: "back" } };
    default:
      return null;
  }
}

function handleRoutine(state: EngineState, command: Command): EngineResult | null {
  const focus = state.routineFocus;

  switch (command) {
    case "up":
      return focus > 0 ? { patch: { routineFocus: focus - 1 } } : {};
    case "down":
      return focus < ROUTINES.length - 1 ? { patch: { routineFocus: focus + 1 } } : {};
    case "go":
      return { patch: { runningRoutine: focus } };
    case "stop":
      return { patch: { runningRoutine: null } };
    // Igual que en Device: "no" siempre significa volver.
    case "no":
      return { nav: { type: "back" } };
    default:
      return null;
  }
}

function handleConfirm(state: EngineState, command: Command): EngineResult | null {
  switch (command) {
    case "yes":
      return {
        patch: { devices: { ...state.devices, lights: false }, brightnessMode: false },
        nav: { type: "back" },
      };
    case "no":
      return { nav: { type: "back" } };
    default:
      return null;
  }
}
