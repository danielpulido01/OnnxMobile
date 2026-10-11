# Descargar el dataset preprocesado

La versión compartida está en el [Release speech-commands-20261008-v1](https://github.com/danielpulido01/OnnxMobile/releases/tag/speech-commands-20261008-v1).
Contiene los tensores originales, sin pérdida de precisión, con 83.150 ejemplos
de entrenamiento, 9.962 de validación y 10.730 de prueba. Las once clases, en orden,
son `yes, no, left, right, up, down, on, off, stop, go, unknown`.

`X_test.npy` contiene los espectrogramas de prueba: `float32`, forma
`(10730, 64, 101, 1)`, disposición NHWC. `y_test.npy` contiene sus etiquetas.
Se reserva test para evaluar los modelos seleccionados por validación.

## Descargar y extraer

Desde la raíz del repositorio, con Python 3.11 y GitHub CLI:

```powershell
gh release download speech-commands-20261008-v1 --repo danielpulido01/OnnxMobile --dir ai/data/releases/speech-commands-20261008-v1
python ai/data/releases/speech-commands-20261008-v1/share_dataset.py unpack --manifest ai/data/releases/speech-commands-20261008-v1/speech-commands-20261008T020826106877Z.bundle.json --output ai/data/processed/speech_commands
```

También se pueden descargar todos los archivos de **Assets** desde el navegador
y colocarlos juntos en `ai/data/releases/speech-commands-20261008-v1/` antes de
ejecutar el segundo comando. Los archivos automáticos `Source code` no contienen
el dataset. El script usa únicamente la biblioteca estándar de Python y verifica
tamaño y SHA-256 del archivo comprimido y de cada archivo extraído. Si el destino
ya existe, se detiene sin sobrescribirlo. Para revisar la descarga sin extraer:

```powershell
python ai/tools/share_dataset.py verify --manifest ai/data/releases/speech-commands-20261008-v1/speech-commands-20261008T020826106877Z.bundle.json
```

Se necesitan unos **5 GB libres** para conservar la descarga y los archivos
extraídos. El resultado queda en:
`ai/data/processed/speech_commands/20261008T020826106877Z/`.

## Entrenar

Instalar las dependencias y configurar W&B como se indica en
[training.md](training.md). Para iniciar o continuar un solo experimento:

```powershell
.\.venv\Scripts\python.exe ai/src/train.py --data ai/data/processed/speech_commands/20261008T020826106877Z --suite ai/runs/speech_commands --device auto --epochs-per-run 5
```

El paquete incluye los seis archivos `X_*.npy`/`y_*.npy`, el manifiesto, la auditoría,
los parámetros de preprocesamiento y `COMPLETED.json`. Los WAV y PNG individuales
se omiten para reducir el tamaño: el entrenamiento lee los tensores directamente.
Las rutas de WAV/PNG en el manifiesto y la ruta de origen en `preprocessing.json`
documentan la exportación original; no hace falta reproducir esas rutas locales
para entrenar. Para explorar el audio o regenerar las imágenes individuales,
ejecutar el notebook de preprocesamiento.

La variante aumentada aplica máscaras de tiempo/frecuencia durante el
entrenamiento, exclusivamente a train; ambos modelos comparten los mismos
conjuntos de validación y prueba. No se distribuye otra copia aumentada estática.

## Publicación y atribución

Los datos grandes se distribuyen mediante Releases y permanecen excluidos de
los commits. El paquete incluye la licencia CC BY 4.0, el README original de
Speech Commands v0.02 y `ATTRIBUTION.md`, que describe la procedencia y las
transformaciones del equipo. Se deben conservar al redistribuirlo.

Para empaquetar una futura exportación completa:

```powershell
python ai/tools/share_dataset.py pack --data ai/data/processed/speech_commands/<version> --output ai/data/releases/<release> --original <carpeta-con-LICENSE-y-README-originales>
```

El empaquetador conserva todos los bytes y genera un manifiesto de checksums.
Si el ZIP supera el límite de un asset de GitHub, produce partes de 1 GiB;
en ese caso se publican las partes y el manifiesto, junto con el script.
