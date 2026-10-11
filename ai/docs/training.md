# Entrenar y evaluar con Weights & Biases

El protocolo completo y los diagramas están en
[architectures.md](architectures.md). La matriz reproducible está en
[experiments.json](../configs/experiments.json). Las redes manuales están en
[models.py](../src/models.py). No hay resultados reales de entrenamiento incluidos
en el repositorio: las tablas e informes se calculan al ejecutar los experimentos.

## Instalar y conectar W&B

Desde `D:\Repos\OnnxMobile`, en PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r ai/requirements.txt
.\.venv\Scripts\wandb.exe login
```

El login solicita la clave fuera del código. También se puede configurar
`WANDB_API_KEY` en el entorno, sin guardarla en Git/notebooks. El proyecto por
defecto es `onnxmobile-speech-commands`; `--entity` permite seleccionar una cuenta
o equipo y `--project` otro nombre. El modo predeterminado es online, para ver
el comportamiento en vivo en W&B. Todas las fases registran runs W&B.

Si el login dice que la clave debe tener 40 caracteres pero la plataforma entrega
una clave de 86 caracteres, comprobar que el cliente instalado sea la versión
fijada actual (0.29.0). El cliente anterior 0.19.11 rechazaba ese formato por su
validación de longitud. Actualizar y pegar la clave completa en el prompt:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade wandb==0.29.0
.\.venv\Scripts\wandb.exe login --relogin --verify
```

Reiniciar el kernel de Jupyter si estaba abierto durante la actualización.
La cuenta personal en CoreWeave Forge usa el login del SDK W&B; no configurar
la página `/wandb/home` como `WANDB_BASE_URL`.

Para la GTX 1650 y un driver compatible con CUDA 12.1 se puede instalar la rueda
CUDA de la misma versión fijada de PyTorch:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Si ya hay instalada una rueda CPU de la misma versión, añadir `--force-reinstall`
para cambiarla por la rueda CUDA. El driver detectado al implementar fue 551.23;
la disponibilidad real se comprueba con `torch.cuda.is_available()`. El CLI usa
CUDA si está disponible y CPU en otro caso; `--device cuda` exige GPU y falla si
no está disponible. Registrar la versión exacta de la rueda con el experimento.

## Un entrenamiento por sesión

Usar una exportación con `COMPLETED.json` y once clases. Por ejemplo:

```powershell
.\.venv\Scripts\python.exe ai/src/train.py --data ai/data/processed/speech_commands/20261008T020826106877Z --suite ai/runs/speech_commands --project onnxmobile-speech-commands --watch
```

Agregar `--entity NOMBRE_EQUIPO` si aplica. Este comando ejecuta **un solo trial**,
de hasta 30 épocas, con validación y early stopping. La misma carpeta `--suite`
reúne los doce experimentos, aunque se ejecuten en días distintos. Sin seleccionar
un nombre, se continúa el primer trial pendiente en el orden de la matriz.
`--workers 0` es portable en Windows;
se puede fijar otro número antes de iniciar la comparación. Los modelos deben
usar el mismo entorno y particiones durante toda la suite.

Para continuar o iniciar el siguiente trial, basta repetir:

```powershell
.\.venv\Scripts\python.exe ai/src/train.py --suite ai/runs/speech_commands --watch
```

Se infieren los datos, configuración, proyecto, entidad, dispositivo y modo W&B
de la suite. Los trials completos se conservan y no se vuelven a entrenar.
Para consultar el estado sin entrenar ni conectar W&B, o elegir un trial:

```powershell
.\.venv\Scripts\python.exe ai/src/train.py --suite ai/runs/speech_commands --list
.\.venv\Scripts\python.exe ai/src/train.py --suite ai/runs/speech_commands --experiment mobilenet_v1-augmented-adam_1e-3 --watch
```

`--list` muestra `complete`, `resumable` o `pending`. Seleccionar un trial completo
lo omite. El flag opcional `--all` permite ejecutar explícitamente todos los
pendientes; no es el comportamiento predeterminado.

Se imprimen la carpeta de la suite y las métricas de cada época.
En W&B, agrupar por el nombre de esa carpeta y filtrar `job_type=train`: estarán los runs
con tags de arquitectura y variante. Graficar `train/loss`, `train_clean/loss`,
`validation/loss`, `train_clean/macro_f1`, `validation/macro_f1` contra `epoch`.
`--watch` agrega histogramas de parámetros/gradientes cada 100 pasos; tiene costo
adicional. Al completar los doce trials, la fase de comparación sube las tablas
y fija los cuatro ganadores. No se hace selección final ni test con una matriz
incompleta. El contrato del modelo usa las clases en `preprocessing.json`, sin
reordenarlas alfabéticamente.

También se puede seguir el flujo en
[02_entrenamiento_evaluacion.ipynb](../notebooks/02_entrenamiento_evaluacion.ipynb).
Ese notebook ejecuta los mismos scripts y muestra sus artefactos; no duplica
rutinas de entrenamiento. Al ejecutar todas sus celdas se entrena un solo trial.
`EXPERIMENT=None` elige el siguiente pendiente; indicar un nombre selecciona uno.
`EPOCHS_PER_RUN` permite pausar por épocas. El test permanece separado con
`RUN_TEST=False` hasta que el usuario lo habilite después de completar la matriz.

## Evaluar los cuatro ganadores

Una vez finalizados los doce trials, reemplazar `<SUITE>` por la carpeta impresa:

```powershell
.\.venv\Scripts\python.exe ai/src/evaluate.py --suite ai/runs/<SUITE>
```

Se cargan exclusivamente los checkpoints elegidos por validación, se evalúan
en el mismo test sin aumento y se registran cuatro runs W&B de test. No se usa
test para early stopping, búsqueda ni para cambiar el ganador. Si el test ya
terminó, el script pide reutilizar los resultados guardados. El informe final
se sube a W&B como artefacto junto con las tablas. La latencia comparativa es
CPU PyTorch, batch=1; no equivale a rendimiento móvil ni incluye preprocesamiento.

## Artefactos locales

| Archivo | Contenido |
| --- | --- |
| `suite.json`, `dataset.json` | Configuración, identidad de código/datos y hashes |
| `<experimento>/config.json` | Hiperparámetros explícitos |
| `<experimento>/environment.json`, `requirements-lock.txt` | Entorno y versiones |
| `<experimento>/architecture.txt` | Capas del modelo real |
| `<experimento>/history.csv`, `learning_curves.png` | Curvas de train, train limpio y validation |
| `<experimento>/last.pt` | Última época completa, modelo actual, optimizador, RNG, historial, paciencia y mejor checkpoint |
| `<experimento>/best.pt` | Modelo con mejor validación, optimizador, clases y pesos |
| `<experimento>/validation_best_confusion.*` | Matriz de validación del mejor checkpoint |
| `training_results.csv` | Resultados de los doce entrenamientos |
| `selection.json` | Cuatro ganadores y mejores global/base, fijados antes de test |
| `<experimento>/test_result.json`, `test_confusion.*` | Métricas por clase y matriz de test |
| `test_results.csv`, `test_results.json`, `report.md` | Comparación e insumos para el informe |
| `wandb/` | Eventos locales del SDK de W&B |

La vista aumentada se genera durante train, con política y RNG propios. No ocupa
una copia completa del dataset; se reconstruye con los tensores originales,
la semilla, la configuración y el código guardados. Validación y test son comunes
a las dos variantes, lo que permite una comparación sin fuga de versiones del
mismo audio a otras particiones.

## Checkpoints, pausas y recuperación

```powershell
.\.venv\Scripts\python.exe ai/src/train.py --suite ai/runs/speech_commands --epochs-per-run 5 --watch
```

Esto ejecuta hasta cinco épocas adicionales de un trial, guarda su progreso y
termina. Repetir el comando continúa ese trial hasta que complete sus 30 épocas
o alcance early stopping; la próxima invocación inicia el siguiente pendiente.
El límite por sesión no cambia el presupuesto total ni los hiperparámetros.

Después de cada época completa se guarda `last.pt`, incluyendo modelo actual,
optimizador, mejor modelo, historial, contadores de early stopping y estados RNG
de Python, NumPy, PyTorch CPU/CUDA, aumento y los tres DataLoaders. Se escribe un
temporal, se vacía a disco y se reemplaza el archivo anterior atómicamente.
Una escritura interrumpida conserva el checkpoint anterior. `best.pt` también
se escribe atómicamente y se puede reconstruir desde `last.pt` al continuar.

Si falla o se interrumpe durante una época, al repetir el comando se recupera
la última época completa y se repite la época en curso. No se promete recuperación
a mitad de un batch ni protección contra pérdida física del disco. Conservar una
copia externa de la carpeta de resultados si se necesita respaldo.

La reanudación exige el mismo código, datos, configuración, versiones, plataforma,
dispositivo, workers y threads para mantener reproducibilidad. Si no coinciden,
se informa el cambio y se conservan los archivos. `--resume-suite` sigue siendo un
alias de `--suite`. Un bloqueo del sistema operativo impide ejecutar dos
entrenamientos simultáneos en la misma suite y se libera si el proceso muere.

Cada continuación registra una nueva sesión W&B en el mismo grupo, con
`resume_from_epoch` y `previous_run_id` para enlazarla con la anterior, tanto online
como offline. El historial local y las curvas finales incluyen todas las épocas;
la tabla de comparación conserva un único resultado por trial completo. Por eso
puede haber más de doce runs W&B si se pausaron o reanudaron experimentos.

Este mecanismo sigue la recomendación de
[PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
de conservar modelo y optimizador para continuar un entrenamiento.

## Trabajar sin conexión

Para máquinas sin Internet, agregar `--wandb-mode offline` al entrenamiento.
La evaluación hereda el mismo modo. Los eventos se guardan mediante el SDK W&B
y se suben después; hasta entonces no hay tablero web. Tras conectar/login:

```powershell
Get-ChildItem -LiteralPath ai/runs/<SUITE>/wandb -Directory -Filter 'offline-run-*' | ForEach-Object { .\.venv\Scripts\python.exe -m wandb sync $_.FullName }
```

Esto sincroniza también los runs de comparación, test e informe, además de los
doce de entrenamiento. No se ofrece un modo sin tracking.

## Verificación técnica sin entrenar el corpus

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ai/tests -v
```

Las pruebas de entrenamiento verifican gradientes de ambas arquitecturas,
parámetros/conectividad de LeNet, repetibilidad, enmascaramiento, ausencia de
aumento en eval, métricas, agregación de loss ponderada, auditoría de hablantes
y selección por validación. La prueba integral ejecuta **12 trials sintéticos de
una época** con `--all` explícito y test de cuatro ganadores con eventos W&B offline.
También compara ambas arquitecturas aumentadas entrenadas de corrido y por
épocas en procesos distintos, comprobando igualdad exacta de pesos y métricas;
verifica que repetir un trial completo no lo modifique, que el modo predeterminado
entrene solo uno y que una escritura fallida preserve el checkpoint previo.
Sus carpetas son temporales: no son resultados experimentales del corpus real.
