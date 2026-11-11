"""Calcula todo el Laboratorio 3 y guarda tablas (CSV/JSON) y figuras (PNG) en results/."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data import load_series, find_series_path, TARGET
from src.pipeline import run_all

RES = ROOT / "results"; FIG = RES / "figures"
C = {"real": "#1f3b4d", "hw": "#e4572e", "lin": "#2a9d8f", "base": "#9aa5b1"}


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=130); plt.close(fig)


def make_figures(d, o):
    sp, p = o["splits"], o["predictions"]
    sel = o["selected"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(d.step, d[TARGET], color=C["real"], lw=1)
    for v, lab in [(sp["train_end"], "fin entrenamiento"), (sp["val_end"], "fin validación")]:
        ax.axvline(v, ls="--", color="#c0392b"); ax.text(v + 3, ax.get_ylim()[1] * .9, lab, fontsize=8)
    ax.set(title="Transacciones por hora y partición cronológica", xlabel="step (hora simulada)", ylabel="transacciones")
    _save(fig, "serie_particion.png")

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    axs[0].hist(d[TARGET], bins=40, color=C["real"]); axs[0].set(title="Distribución del objetivo", xlabel="transacciones")
    prof = d[d.step <= sp["train_end"]].groupby("fase_24")[TARGET].mean()
    axs[1].plot(prof.index, prof.values, marker="o", color=C["hw"]); axs[1].set(title="Perfil por fase de 24 h (train)", xlabel="step % 24")
    a = o["acf"]; axs[2].bar(a.lag[1:73], a.acf[1:73], color=C["lin"]); axs[2].set(title="ACF (hasta 72 h)", xlabel="rezago")
    _save(fig, "eda.png")

    te = p[p.bloque == "test"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(te.step, te.real, color=C["real"], lw=1.6, label="Real")
    ax.plot(te.step, te[sel["holt_winters"]], color=C["hw"], label=f"{sel['holt_winters']} (1 h)")
    ax.plot(te.step, te[sel["lineal"]], color=C["lin"], label=f"{sel['lineal']} (1 h)")
    ax.plot(te.step, te[sel["holt_winters"] + "_origen_fijo"], color=C["hw"], ls=":", label="Holt-Winters origen fijo (144 h)")
    ax.legend(fontsize=8); ax.set(title="Prueba: real vs pronósticos", xlabel="step", ylabel="transacciones")
    _save(fig, "test_pronosticos.png")

    t = o["test"].sort_values("MAE")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = [C["hw"] if f == "holt_winters" else C["lin"] if f == "regresion_lineal" else C["base"] for f in t.familia]
    ax.barh(t.modelo[::-1], t.MAE[::-1], color=cols[::-1]); ax.set(title="MAE en prueba (1 h)", xlabel="MAE")
    _save(fig, "comparacion_mae.png")

    hw = o["hw_final"]; L, B, S = hw.components(d[TARGET].to_numpy())
    fig, axs = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axs[0].plot(d.step, L, color=C["hw"]); axs[0].set_title("Holt-Winters: nivel filtrado")
    axs[1].plot(d.step, S, color=C["lin"]); axs[1].set_title("Holt-Winters: componente estacional"); axs[1].set_xlabel("step")
    _save(fig, "hw_componentes.png")

    f = o["forecast_72"]
    fig, ax = plt.subplots(figsize=(11, 3.6)); h = d.tail(168)
    ax.plot(h.step, h[TARGET], color=C["real"], label="Observado (últimas 168 h)")
    ax.plot(f.step, f.pronostico_HW, color=C["hw"], label="Pronóstico HW 72 h"); ax.legend()
    ax.set(title="Pronóstico Holt-Winters a 72 horas (modelo ajustado con toda la serie)", xlabel="step")
    _save(fig, "hw_pronostico_72h.png")


def run(path=None):
    RES.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    d = load_series(path); o = run_all(d)
    for k in ["splits_table", "descriptive", "regimes", "validation", "hw_grid_validation", "test", "test_origen_fijo",
              "cv", "hypotheses", "coef_hac", "ljungbox_hw", "hw_params", "forecast_72", "predictions", "acf"]:
        o[k].to_csv(RES / f"{k}.csv", index=False)
    summary = {"fuente": str(find_series_path() if path is None else path), "n_horas": len(d),
               "n_transacciones": int(d[TARGET].sum()), "splits": o["splits"], "seleccion": o["selected"],
               "acf": o["acf_resumen"]}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(d, o)
    return d, o


if __name__ == "__main__":
    d, o = run()
    pd.options.display.width = 160
    print(o["selected"]); print(o["test"][["modelo", "MAE", "RMSE", "R2"]].round(3).to_string(index=False))
