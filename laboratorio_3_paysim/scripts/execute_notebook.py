"""Ejecuta notebooks/Laboratorio_3.ipynb SIN Jupyter y guarda las salidas (texto, tablas y figuras) en el propio .ipynb.
Uso:  python scripts/execute_notebook.py      (equivale a 'Run All' + guardar)."""
from __future__ import annotations
import ast, base64, contextlib, io, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NB = ROOT / "notebooks" / "Laboratorio_3.ipynb"


class Markdown:
    def __init__(self, s): self.s = s


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    os.chdir(ROOT / "notebooks")
    g, count = {"__name__": "__main__"}, 0

    def to_outputs(obj, outs, exec_count=None):
        if obj is None:
            return
        if isinstance(obj, Markdown):
            data = {"text/markdown": obj.s, "text/plain": obj.s}
        elif isinstance(obj, pd.DataFrame):
            data = {"text/html": obj.to_html(max_rows=40, max_cols=30), "text/plain": obj.to_string(max_rows=40, max_cols=30)}
        elif isinstance(obj, pd.Series):
            data = {"text/plain": obj.to_string()}
        else:
            data = {"text/plain": repr(obj) if not isinstance(obj, str) else obj}
        if exec_count is None:
            outs.append({"output_type": "display_data", "data": data, "metadata": {}})
        else:
            outs.append({"output_type": "execute_result", "data": data, "metadata": {}, "execution_count": exec_count})

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        src = "".join(cell["source"]); outs = []
        buf = io.StringIO()

        def show():
            for num in plt.get_fignums():
                fig = plt.figure(num); b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
                outs.append({"output_type": "display_data", "metadata": {},
                             "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}})
            plt.close("all")
        def _disp(*a, _o=outs):
            for x in a:
                to_outputs(x, _o)
        g["__display"] = _disp
        plt.show = show
        tree = ast.parse(src)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        g["display"], g["Markdown"] = g["__display"], Markdown       # salidas de ESTA celda
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, "<celda>", "exec"), g)
            g["display"], g["Markdown"] = g["__display"], Markdown   # sustituye el respaldo del import de IPython
            if last is not None:
                val = eval(compile(ast.Expression(last.value), "<celda>", "eval"), g)
                to_outputs(val, outs, exec_count=count)
        show()
        text = buf.getvalue()
        if text:
            outs.insert(0, {"output_type": "stream", "name": "stdout", "text": text.splitlines(True)})
        cell["outputs"], cell["execution_count"] = outs, count
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook ejecutado y guardado con salidas: {NB}")


if __name__ == "__main__":
    main()
