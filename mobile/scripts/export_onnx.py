"""Exporta el checkpoint MobileNetV1 a ONNX para la app y verifica el resultado.

Uso (desde mobile/scripts):
    .\\.venv\\Scripts\\python.exe export_onnx.py
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[1]
EXPERIMENT = SCRIPTS / "input" / "mobilenet_v1-augmented-adam_b32_e15"
OUTPUT = REPO / "mobile" / "assets" / "model"
INPUT_SHAPE = (1, 1, 64, 101)  # NCHW: batch, canal, bandas Mel, frames
OPSET = 17

# Se importa la arquitectura del equipo de entrenamiento sin copiarla.
sys.path.insert(0, str(REPO / "ai" / "src"))
from models import build_model  # noqa: E402


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_preprocessing(dataset):
    # Misma serialización que el notebook 01 en Windows: indent=2, ASCII y CRLF.
    text = json.dumps(dataset["metadata"], indent=2, ensure_ascii=True).replace("\n", "\r\n")
    path = OUTPUT / "preprocessing.json"
    path.write_bytes(text.encode("utf-8"))  # write_bytes evita convertir \r\n en \r\r\n
    digest = sha256(path.read_bytes())
    expected = dataset["hashes"]["preprocessing.json"]
    print(f"preprocessing.json SHA-256: {digest}")
    print(f"esperado (dataset.json):    {expected}")
    if digest != expected:
        raise ValueError("preprocessing.json no coincide con el hash registrado")
    return path


def load_model():
    checkpoint = torch.load(EXPERIMENT / "best.pt", map_location="cpu", weights_only=True)
    config, classes = checkpoint["config"], checkpoint["classes"]
    model = build_model(config["architecture"], len(classes), config["width_mult"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # BatchNorm con estadísticas de entrenamiento
    return model, classes


def export(model, path):
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    torch.onnx.export(model, (dummy,), str(path), input_names=["input"], output_names=["logits"],
                      opset_version=OPSET, dynamo=False)  # sin dynamic_axes: tamaño fijo
    model_proto = onnx.load(str(path))
    onnx.checker.check_model(model_proto)
    opsets = {o.domain or "ai.onnx": o.version for o in model_proto.opset_import}
    print(f"opset: {opsets}")
    if opsets.get("ai.onnx") != OPSET:
        raise ValueError(f"Se esperaba opset {OPSET}")
    extra = sorted(p.name for p in path.parent.iterdir() if p.name not in ("model.onnx", "preprocessing.json"))
    if extra:
        raise ValueError(f"Archivos inesperados junto al modelo (¿pesos externos?): {extra}")


def check_session(session, classes):
    (inp,), (out,) = session.get_inputs(), session.get_outputs()
    print(f"entrada ONNX: {inp.name} {inp.shape} {inp.type}")
    print(f"salida ONNX:  {out.name} {out.shape} {out.type}")
    if inp.name != "input" or out.name != "logits":
        raise ValueError("Nombres de entrada/salida incorrectos")
    if inp.shape != list(INPUT_SHAPE) or out.shape != [1, len(classes)]:
        raise ValueError("Formas de entrada/salida incorrectas")


def check_parity(model, session):
    generator = np.random.default_rng(42)
    inputs = {f"aleatoria_{i}": generator.random(INPUT_SHAPE, dtype=np.float32) for i in range(5)}
    inputs["ceros"] = np.zeros(INPUT_SHAPE, dtype=np.float32)
    inputs["unos"] = np.ones(INPUT_SHAPE, dtype=np.float32)
    worst = 0.0
    for name, x in inputs.items():
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        actual = session.run(["logits"], {"input": x})[0]
        diff = float(np.abs(expected - actual).max())
        same = int(expected.argmax()) == int(actual.argmax())
        print(f"  {name:12s} dif. máx = {diff:.3e}  argmax igual = {same}")
        if not same:
            raise ValueError(f"La clase predicha difiere en {name}")
        worst = max(worst, diff)
    print(f"diferencia máxima absoluta PyTorch vs ONNX Runtime: {worst:.3e}")
    if worst > 1e-4:
        raise ValueError("Diferencia demasiado grande")


def measure_latency(session, runs=100, warmup=10):
    x = np.random.default_rng(0).random(INPUT_SHAPE, dtype=np.float32)
    for _ in range(warmup):
        session.run(["logits"], {"input": x})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(["logits"], {"input": x})
        times.append((time.perf_counter() - start) * 1000)
    print(f"latencia ONNX Runtime CPU ({runs} corridas, {warmup} de calentamiento): "
          f"promedio {statistics.mean(times):.3f} ms, mediana {statistics.median(times):.3f} ms")
    print("  (result.json reporta 8.25 ms, pero es PyTorch en CUDA: no es comparable)")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((EXPERIMENT / "dataset.json").read_text(encoding="utf-8"))
    print(f"torch {torch.__version__} | onnx {onnx.__version__} | onnxruntime {ort.__version__}")
    write_preprocessing(dataset)

    model, classes = load_model()
    if classes != dataset["metadata"]["classes"]:
        raise ValueError("Las clases del checkpoint no coinciden con preprocessing.json")
    path = OUTPUT / "model.onnx"
    export(model, path)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    check_session(session, classes)
    check_parity(model, session)

    print(f"parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"tamaño de model.onnx: {path.stat().st_size:,} bytes ({path.stat().st_size / 1024:.1f} KiB)")
    measure_latency(session)
    print("orden de etiquetas (índice del logit -> clase):")
    for i, label in enumerate(classes):
        print(f"  {i:2d} {label}")
    print(f"listo: {path}")


if __name__ == "__main__":
    main()
