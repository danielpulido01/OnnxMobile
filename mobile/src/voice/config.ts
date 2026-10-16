// Umbral de confianza para aceptar un comando: se pasa a classify() de labels.ts.
// - Elegido con scripts/calibrate_threshold.mjs sobre 2000 audios de validación.
// - Compromiso: ~92 % de comandos aceptados, ~2.3 % cambiados por otro comando y
//   ~1.7 % de "unknown" tomados como comando.
// - Debe revisarse con grabaciones reales del micrófono del celular.
export const CONFIDENCE_THRESHOLD = 0.7;

// SOLO DESARROLLO: URL de scripts/dev_server.mjs que usa RemoteClassifier, p. ej.
// "http://192.168.1.50:8787". Se define en mobile/.env.local (ver .env.example).
// El objetivo final es OnnxClassifier en Android, que no necesita servidor.
export const DEV_SERVER_URL = process.env.EXPO_PUBLIC_DEV_SERVER_URL ?? "";
