"""Pruebas de coherencia: fuga de información, fórmulas de métricas, Holt-Winters y particiones."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.data import load_series, make_splits, TARGET
from src.holtwinters import HoltWinters, HWConfig
from src.modeling import metrics, make_features, ols_hac, holm, acf, ljung_box, decompose


def main():
    d = load_series(); y = d[TARGET].to_numpy(float); sp = make_splits(d)
    # 1) Partición cronológica y sin solapamiento
    assert sp["first_supervised"] == 25 and sp["train_end"] < sp["val_end"] < sp["last"]
    x = make_features(d)
    assert x.step.min() == 25 and (x.lag_1.values[1:] == x[TARGET].values[:-1]).all()
    assert (x.lag_24.dropna().values == d[TARGET].values[:len(x)].astype(float)[:len(x.lag_24.dropna())]).all()
    # 2) Fórmulas de métricas
    a, b = np.array([10., 20, 0, 40]), np.array([12., 18, 5, 30])
    m = metrics(a, b)
    assert np.isclose(m["MAE"], np.mean(np.abs(a - b))) and np.isclose(m["RMSE"], np.sqrt(np.mean((a - b) ** 2)))
    assert np.isclose(m["MAPE_pct"], 100 * np.mean(np.abs(a[[0, 1, 3]] - b[[0, 1, 3]]) / a[[0, 1, 3]]))
    assert np.isclose(m["R2"], 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())
    # 3) Holt-Winters: sin fuga (cambiar el futuro no altera la predicción de t) y coherencia forecast/one_step
    for cfg in [HWConfig(None, False, "add"), HWConfig("add", True, "mul"), HWConfig(None, False, "add", 24, "log1p")]:
        hw = HoltWinters(cfg).fit(y[:300])
        p1 = hw.one_step(y)
        y2 = y.copy(); y2[400:] *= 7
        assert np.allclose(p1[:400], hw.one_step(y2)[:400]), "fuga en one_step"
        assert np.isclose(hw.forecast(1, y[:400])[0], p1[400]), "forecast(1) debe igualar one_step"
        assert np.isfinite(hw.forecast(72)).all() and (hw.forecast(72) >= 0).all()
        assert np.allclose(hw.fitted_, HoltWinters(cfg).fit(y[:300]).fitted_)
    # 4) HAC / Holm / ACF / Ljung-Box / descomposición
    rng = np.random.default_rng(0); X = rng.normal(size=(300, 2)); yy = 1 + X @ [2, 0] + rng.normal(size=300)
    beta, V, _ = ols_hac(yy, X, 5); assert np.allclose(beta, [1, 2, 0], atol=.4) and (np.diag(V) > 0).all()
    assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06])
    assert np.isclose(acf(yy, 3)[0], 1)
    q, p = ljung_box(rng.normal(size=500), 10); assert 0 <= p <= 1
    dc = decompose(y[:240]); assert np.allclose((dc.tendencia + dc.estacionalidad + dc.residuo).dropna(), dc.observado[dc.tendencia.notna()])
    print("VERIFICACIÓN OK: particiones, métricas, Holt-Winters sin fuga, HAC/Holm/Ljung-Box y descomposición.")


if __name__ == "__main__":
    main()
