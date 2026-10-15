// Los 10 comandos de Speech Commands v0.02 que entiende la app.
export const COMMANDS = [
  "yes",
  "no",
  "up",
  "down",
  "left",
  "right",
  "on",
  "off",
  "stop",
  "go",
] as const;

export type Command = (typeof COMMANDS)[number];
