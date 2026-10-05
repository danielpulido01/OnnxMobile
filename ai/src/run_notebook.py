"""Ejecuta el notebook con el Python actual y conserva las salidas, incluso al fallar."""
import argparse
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def run(output=None):
    ai_dir = Path(__file__).resolve().parents[1]
    source = ai_dir / "notebooks" / "01_exploracion_speech_commands.ipynb"
    target = Path(output) if output else source.with_name(source.stem + ".executed.ipynb")
    target.parent.mkdir(parents=True, exist_ok=True)
    for name, folder in [("JUPYTER_RUNTIME_DIR", "jupyter-runtime"),
                         ("IPYTHONDIR", "ipython"), ("MPLCONFIGDIR", "matplotlib")]:
        os.environ.setdefault(name, str(ai_dir / ".cache" / folder))
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=None,
                            resources={"metadata": {"path": str(ai_dir)}})

    def report(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Ejecutando celda {cell_index + 1}/{len(notebook.cells)}", flush=True)

    client.on_cell_start = report
    try:
        client.execute()
    finally:
        nbformat.write(notebook, target)
        print(f"Notebook con salidas: {target}", flush=True)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Destino del notebook ejecutado")
    run(parser.parse_args().output)
