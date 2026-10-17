// Umbral de confianza para aceptar un comando: se pasa a classify() de labels.ts.
// - Elegido con scripts/calibrate_threshold.mjs sobre 2000 audios de validación.
// - Compromiso: ~92 % de comandos aceptados, ~2.3 % cambiados por otro comando y
//   ~1.7 % de "unknown" tomados como comando.
// - Debe revisarse con grabaciones reales del micrófono del celular.
export const CONFIDENCE_THRESHOLD = 0.7;

// Filtro de silencio: si el pico (máximo |muestra|) de la ventana de 1 s es menor, no se ejecuta
// el modelo y es un no-op ("Sin voz"). Hace falta porque el espectrograma se normaliza con el
// máximo del clip (ref = max) y el silencio amplificado llegaba a clasificarse como comando
// (p. ej. pico 0.00019 -> "up" 0.79). Valor inicial; ajustar con los picos de los logs [capture].
export const SILENCE_PEAK_THRESHOLD = 0.02;
