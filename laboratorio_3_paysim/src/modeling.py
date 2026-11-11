"""Regresión lineal supervisada, baselines, métricas, pruebas de hipótesis (HAC) y utilidades de series."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data import TARGET, M

LAGS = ["lag_1", "lag_24"]
CALENDAR = ["step", "sin24", "cos24", "sin12", "cos12"]

# ------------------------------------------------------------------ métricas
def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    nz = y != 0
    return {"MAE": float(mean_absolute_error(y, yhat)),
            "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
            "MAPE_pct": float(100 * np.mean(np.abs(y[nz] - yhat[nz]) / np.abs(y[nz]))) if nz.any() else np.nan,
            "WAPE_pct": float(100 * np.abs(y - yhat).sum() / np.abs(y).sum()),
            "R2": float(r2_score(y, yhat)), "n": int(len(y))}

# ------------------------------------------------------------------ variables
def make_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d[["step", TARGET]].copy()
    x["lag_1"] = x[TARGET].shift(1)
    x["lag_24"] = x[TARGET].shift(24)
    x["lag_168"] = x[TARGET].shift(168)
    x["media_estacional_7"] = pd.concat([x[TARGET].shift(24 * k) for k in range(1, 8)], axis=1).mean(axis=1)
    ang = 2 * np.pi * x["step"] / 24
    x["sin24"], x["cos24"] = np.sin(ang), np.cos(ang)
    x["sin12"], x["cos12"] = np.sin(2 * ang), np.cos(2 * ang)
    return x[x.step >= M + 1].reset_index(drop=True)          # se pierden las primeras 24 h (sin rezago_24)


def build_pipeline(cols: list[str]) -> Pipeline:
    """selección de columnas -> StandardScaler -> LinearRegression (el escalador se ajusta solo con train)."""
    ct = ColumnTransformer([("sc", StandardScaler(), cols)], remainder="drop")
    return Pipeline([("prep", ct), ("lm", LinearRegression())])


def fit_linear(train: pd.DataFrame, cols) -> Pipeline:
    return build_pipeline(cols).fit(train[cols], train[TARGET])


def predict_linear(model: Pipeline, df: pd.DataFrame, cols) -> np.ndarray:
    return np.maximum(model.predict(df[cols]), 0.0)          # regla fija: un conteo no es negativo


def linear_coefficients(model: Pipeline, cols) -> pd.DataFrame:
    sc, lm = model.named_steps["prep"].named_transformers_["sc"], model.named_steps["lm"]
    raw = lm.coef_ / sc.scale_
    return pd.DataFrame({"variable": cols, "coef_estandarizado": lm.coef_, "coef_unidades_originales": raw})


def baseline_predictions(x: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Referencias simples; todas usan solo información anterior a t (media_train usa solo train)."""
    return pd.DataFrame({"media_train": train[TARGET].mean(), "persistencia_1": x["lag_1"],
                         "estacional_24": x["lag_24"], "estacional_168": x["lag_168"],
                         "media_estacional_7": x["media_estacional_7"]}, index=x.index)

# ------------------------------------------------------------------ Newey-West (HAC) + Wald
def ols_hac(y, X, maxlags: int = 24):
    """OLS con covarianza HAC Bartlett (Newey-West) y corrección n/(n-k)."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, maxlags + 1):
        G = Xu[l:].T @ Xu[:-l]
        S += (1 - l / (maxlags + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi * n / (n - k)
    return beta, V, u


def wald(beta, V, idx):
    idx = list(idx)
    b = beta[idx]
    stat = float(b @ np.linalg.solve(V[np.ix_(idx, idx)], b))
    return stat, float(stats.chi2.sf(stat, len(idx))), len(idx)


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def acf(y, nlags: int):
    y = np.asarray(y, float) - np.mean(y)
    den = float(y @ y)
    if den == 0:
        return np.r_[1.0, np.full(nlags, np.nan)]
    return np.array([1.0] + [float(y[k:] @ y[:-k]) / den if k < len(y) else np.nan for k in range(1, nlags + 1)])


def ljung_box(resid, lags: int = 24):
    r = np.asarray(resid, float); r = r[np.isfinite(r)]; n = len(r)
    rho = acf(r, lags)[1:]
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def decompose(y, m: int = M, model: str = "additive") -> pd.DataFrame:
    """Descomposición clásica (media móvil centrada 2xm). `model`: 'additive' | 'multiplicative'."""
    s = pd.Series(np.asarray(y, float))
    w = np.r_[0.5, np.ones(m - 1), 0.5] / m
    trend = s.rolling(m + 1, center=True).apply(lambda v: float(v @ w), raw=True)
    det = s - trend if model == "additive" else s / trend.replace(0, np.nan)
    seas = det.groupby(np.arange(len(s)) % m).transform("mean")
    seas = seas - seas.mean() if model == "additive" else seas / seas.mean()
    res = s - trend - seas if model == "additive" else s / (trend * seas)
    return pd.DataFrame({"observado": s, "tendencia": trend, "estacionalidad": seas, "residuo": res})
