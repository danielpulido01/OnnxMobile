# Preparación de Speech Commands para una CNN

El notebook [01_exploracion_speech_commands.ipynb](notebooks/01_exploracion_speech_commands.ipynb)
documenta en español la exploración y el procesamiento de
[`yashdogra/speech-commands`](https://www.kaggle.com/datasets/yashdogra/speech-commands).
Prepara **11 clases**: `yes, no, left, right, up, down, on, off, stop, go, unknown`,
en ese orden (IDs 0–10). `unknown` agrupa todas las otras palabras del corpus,
conservando su palabra original en `source_label`. `_background_noise_` se excluye:
contiene ruido, no palabras. Se conservan todos los audios válidos y únicos,
sin submuestreo de la categoría nueva.
El entrenamiento y la exportación ONNX quedan para una etapa posterior.

Corpus original: Speech Commands v0.02, Pete Warden / Google, licencia CC BY 4.0.
Conservar la atribución al compartir los datos derivados; los archivos `README.md`
y `LICENSE` originales acompañan la descarga.

## Ejecutar

Desde la raíz del repositorio, con Python 3.11, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ai/requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab ai/notebooks/01_exploracion_speech_commands.ipynb
```

Seleccionar el kernel de este entorno y ejecutar todas las celdas en orden.
La primera ejecución requiere Internet, varios GB de disco para descargar y
descomprimir el dataset y espacio adicional para imágenes y tensores.
También se puede ejecutar sin abrir Jupyter:

```powershell
.\.venv\Scripts\python.exe ai/src/run_notebook.py
```

Este comando usa el mismo intérprete como kernel y guarda las salidas en
`ai/notebooks/01_exploracion_speech_commands.executed.ipynb`. Si una celda falla,
conserva el error y las salidas anteriores para diagnosticarlo.

La primera descarga resuelve la última versión disponible; `dataset_lock.json`
registra esa versión para reutilizarla. Para cambiarla deliberadamente, editar
`DATASET_VERSION` en la celda de configuración. Las credenciales, si Kaggle las
solicita, se configuran fuera del notebook.

Para utilizar una descarga ya existente y otra carpeta de salida:

```powershell
$env:SPEECH_COMMANDS_DATA_DIR = 'D:\datos\speech_commands'
$env:SPEECH_COMMANDS_OUTPUT_DIR = 'D:\datos\speech_commands_procesado'
.\.venv\Scripts\python.exe ai/src/run_notebook.py
```

La ruta de entrada puede contener directorios intermedios, pero debe identificar
una sola colección de carpetas de comandos. Los resultados se crean en una carpeta
nueva por ejecución; nunca se mezclan imágenes de ejecuciones anteriores.

## Qué queda registrado

1. Versiones del entorno y procedencia del dataset.
2. Inventario de palabras y asignación de los diez comandos más `unknown`.
3. Conteos, hablantes, duraciones, muestreo, canales, amplitud y calidad de audio.
4. Exclusiones justificadas de archivos inválidos, silencio exacto y duplicados.
5. Particiones oficiales, o particiones deterministas por hablante si faltan las listas.
6. Formas de onda, reproducción de ejemplos y espectrogramas de entrenamiento.
7. Conversión de todos los ejemplos conservados y exportación de imágenes y tensores.
8. Verificaciones de forma, rango, etiquetas, PNG y ausencia de fuga entre particiones.
9. Conclusiones calculadas sobre los datos de la ejecución.

## Representación y artefactos

Audio mono a 16 kHz y un segundo, con padding al final o recorte central;
espectrograma log-Mel de **64 × 101**, un canal, normalizado a `[0, 1]` respecto
del máximo del clip y un rango de 80 dB. Los PNG contienen únicamente píxeles de
intensidad: sin ejes, rótulos ni mapas de color. Cada fila del PNG corresponde a
la misma fila Mel del tensor (la fila cero contiene las frecuencias más bajas).

En `ai/data/processed/speech_commands/<fecha-UTC>/`:

| Archivo | Contenido |
| --- | --- |
| `images/<split>/<label>/*.png` | Imágenes de un canal, 8 bits |
| `X_train.npy`, `X_validation.npy`, `X_test.npy` | `float32`, forma `(N, 64, 101, 1)` |
| `y_train.npy`, `y_validation.npy`, `y_test.npy` | Etiquetas enteras `int64` |
| `manifest.csv` | WAV, PNG, clase final, palabra original (`source_label`), hablante, partición, índice y hashes |
| `audio_audit.csv` | Mediciones y motivo de cada exclusión |
| `inventory.csv`, `split_counts.csv` | Conteos de origen y de particiones |
| `unknown_word_counts.csv` | Conteos por palabra original dentro de `unknown` y por partición |
| `preprocessing.json` | Parámetros, clases, procedencia y entorno |
| `requirements-lock.txt` | Versiones instaladas, incluidas las transitivas |
| `summary.md` | Conclusiones calculadas |
| `COMPLETED.json` | Marca de exportación verificada y hash del manifiesto |

Usar solo carpetas con `COMPLETED.json`. Los tensores se pueden leer con
`np.load(path, mmap_mode="r", allow_pickle=False)` sin cargarlos completos en RAM.
Para PyTorch, convertir NHWC a NCHW mediante `np.transpose(X, (0, 3, 1, 2))`.
El orden de clases está en `preprocessing.json`; un cargador que ordene carpetas
alfabéticamente debe ajustar sus etiquetas a ese contrato.

La normalización no estima estadísticas del conjunto. El notebook no aplica
aumento de datos; cuando se agregue, deberá afectar únicamente entrenamiento.
La clase `unknown` aprende de las otras palabras incluidas en el corpus; esto no
garantiza rechazar cualquier palabra nunca vista. Silencio y ruido no se mezclan
con palabras y requieren tratamiento propio para un reconocedor continuo.

Al agrupar todas las demás palabras, `unknown` queda mucho más grande que cada
comando. El notebook muestra ese desbalance, sin descartar ejemplos para igualar
conteos. Durante el entrenamiento se podrán usar pesos de clase o muestreo,
calculados solo con entrenamiento; reportar métricas por clase y macro-F1.

Los PNG se nombran `<palabra_original>__<nombre_WAV>.png` para evitar colisiones
al agrupar palabras con nombres WAV iguales en `images/<split>/unknown/`.
El procesamiento por lotes usa cuatro trabajadores para acelerar la lectura y
escritura del corpus completo sin cambiar el orden de los ejemplos; `WORKERS = 1`
en la configuración del notebook permite ejecutarlo secuencialmente.

## Verificación sin descargar el dataset

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ai/tests -v
```

La prueba ejecuta las celdas con WAV sintéticos locales, incluidos audio corto,
largo, estéreo, a 8 kHz, silencio, duplicados, etiquetas contradictorias y un WAV
dañado. Verifica ambas estrategias de partición, la agrupación de varias palabras
en `unknown`, la exclusión del ruido, la conservación de nombres WAV coincidentes
y la detección de fuga de hablantes, también para `unknown`. Los datos sintéticos
no se presentan como hallazgos del dataset real.

Los datos, cachés, entorno virtual y notebook ejecutado se excluyen de Git.

## Resultados con once clases

La auditoría de la versión 1 de Kaggle seleccionó las 105.829 grabaciones de
palabras y excluyó los seis archivos de `_background_noise_`. Después de eliminar
1.987 copias de audio idénticas, quedan **103.842 ejemplos**.

`unknown` reúne **66.133 grabaciones de 25 palabras** (67.283 antes de la limpieza).
Las particiones oficiales, sin hablantes compartidos, quedan así:

| Grupo | Entrenamiento | Validación | Prueba | Total |
| --- | ---: | ---: | ---: | ---: |
| Diez comandos | 30.064 | 3.692 | 3.953 | 37.709 |
| `unknown` | 53.086 | 6.270 | 6.777 | 66.133 |
| Total | 83.150 | 9.962 | 10.730 | 103.842 |

El notebook ejecutado guarda los gráficos, tablas y el directorio de los nuevos
artefactos. La configuración de esa carpeta debe indicar once elementos en
`classes`, incluido `unknown` con ID 10.

## Exploración anterior (diez clases, antes de incluir unknown)

Los siguientes conteos describen la exportación anterior de diez clases. Para
las once clases, consultar `summary.md`, `split_counts.csv` y `preprocessing.json`
de la nueva carpeta generada por el notebook. No mezclar ambas exportaciones.

En la versión 1 de Kaggle (corpus v0.02) se encontraron 38.546 WAV de las diez
clases. Todos son mono a 16 kHz; duran entre 0,298625 y 1 segundo. Se excluyeron
837 copias de audio idénticas, conservando 37.709 grabaciones. No se encontraron
WAV inválidos, silencio digital exacto ni duplicados con etiquetas contradictorias.

Las listas oficiales dejan 30.064 ejemplos de entrenamiento, 3.692 de validación
y 3.953 de prueba después de la limpieza, sin hablantes compartidos entre
particiones. Estos conteos se obtuvieron del notebook; pueden cambiar si se
elige otra versión o política de limpieza.
