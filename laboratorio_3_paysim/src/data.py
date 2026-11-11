"""Lectura de la serie horaria PaySim exactamente como en el Laboratorio 2.

El Laboratorio 2 trabajó con `results/hourly_series.csv`
(columnas: step, transacciones_hora, monto_total, monto_promedio, monto_mediano, hora_dia).
Orden de búsqueda: variable de entorno LAB_SERIES -> results/hourly_series.csv -> data/hourly_series.csv
-> si existe data/transacciones.csv (CSV transaccional del Lab 1) se agrega por `step`.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "transacciones_hora"
M = 24  # periodo estacional (horas)


def find_series_path() -> Path | None:
    env = os.environ.get("LAB_SERIES")
    cands = [Path(env)] if env else []
    cands += [ROOT / "results/hourly_series.csv", ROOT / "data/hourly_series.csv", ROOT / "data/transacciones.csv"]
    return next((p for p in cands if p.exists()), None)


def _aggregate_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["step", "monto"])
    g = df.groupby("step")["monto"]
    return pd.DataFrame({"step": g.size().index, TARGET: g.size().values, "monto_total": g.sum().values,
                         "monto_promedio": g.mean().values, "monto_mediano": g.median().values}).reset_index(drop=True)


def load_series(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else find_series_path()
    if p is None:
        raise FileNotFoundError(
            "No se encontró la serie horaria. Copia results/hourly_series.csv del Laboratorio 2 en "
            "'results/' o 'data/' de este proyecto (o define LAB_SERIES con su ruta).")
    d = _aggregate_transactions(p) if p.name == "transacciones.csv" else pd.read_csv(p)
    need = {"step", TARGET}
    if not need <= set(d.columns):
        raise ValueError(f"Faltan columnas {need - set(d.columns)} en {p}")
    d = d.sort_values("step").reset_index(drop=True)
    if not np.isfinite(d[TARGET]).all() or (d[TARGET] < 0).any():
        raise ValueError("La serie contiene valores no finitos o negativos.")
    if not d["step"].diff().dropna().eq(1).all() or d["step"].duplicated().any():
        raise ValueError("`step` debe ser consecutivo y sin duplicados (frecuencia horaria regular).")
    d["fase_24"] = d["step"] % M                          # hora del ciclo diario simulado
    d["ciclo"] = (d["step"] - 1) // M + 1                 # número de ciclo de 24 h ("día" simulado)
    return d


def make_splits(d: pd.DataFrame, first_supervised: int = M + 1) -> dict:
    """Partición cronológica 60/20/20 sobre las filas supervisadas (steps >= 25), igual que el Lab 3 previo:
    con 743 horas -> train 25-455, validación 456-599, prueba 600-743. Holt-Winters usa además 1-24 en train."""
    steps = d.loc[d.step >= first_supervised, "step"].to_numpy()
    n = len(steps)
    a, b = int(n * .6), int(n * .8)
    return {"train_end": int(steps[a - 1]), "val_end": int(steps[b - 1]), "last": int(steps[-1]),
            "first_supervised": int(steps[0])}


def block_of(step: int, sp: dict) -> str:
    return "train" if step <= sp["train_end"] else ("validacion" if step <= sp["val_end"] else "test")
