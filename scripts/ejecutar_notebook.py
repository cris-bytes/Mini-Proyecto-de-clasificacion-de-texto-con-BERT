"""Ejecuta la entrega completa en un kernel nuevo del Python activo."""
import ast
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
# Kernel aislado: no depende de kernels registrados en la cuenta del usuario.
kernel_dir = ROOT / ".cache" / "jupyter" / "kernels" / "massive"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "MASSIVE (entorno activo)", "language": "python",
}), encoding="utf-8")
os.environ["JUPYTER_PATH"] = str(kernel_dir.parents[1])
os.environ["IPYTHONDIR"] = str(ROOT / ".cache" / "ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".cache" / "jupyter-runtime")
notebook_path = ROOT / "clasificacion_intenciones_beto.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
nbformat.validate(notebook)
retraining = None
for cell in notebook.cells:
    if cell.cell_type == "code":
        for statement in ast.parse(cell.source).body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "REENTRENAR"
                for target in statement.targets
            ):
                retraining = ast.literal_eval(statement.value)
if retraining is not True:
    raise ValueError("La verificación completa requiere REENTRENAR = True en el notebook.")
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

def progress(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Ejecutando celda {cell_index + 1}/{len(notebook.cells)}", flush=True)

start = time.perf_counter()
client = NotebookClient(notebook, timeout=7200, kernel_name="massive",
                        resources={"metadata": {"path": str(ROOT)}},
                        on_cell_start=progress)
try:
    client.execute()
except Exception:
    # Conserva el notebook entregado; deja el fallo disponible para diagnóstico.
    nbformat.write(notebook, ROOT / ".cache" / "ejecucion_fallida.ipynb")
    raise
nbformat.write(notebook, notebook_path)
record = {"estado": "completado", "kernel_nuevo": True,
          "celdas_codigo": sum(c.cell_type == "code" for c in notebook.cells),
          "segundos": time.perf_counter() - start,
          "python": sys.version.split()[0],
          "reentrenamiento": True}
(ROOT / "resultados" / "ejecucion_completa.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2), flush=True)
