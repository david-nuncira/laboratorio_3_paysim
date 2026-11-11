"""Holt-Winters (suavizamiento exponencial triple) implementado desde cero.

Formulación en forma de componentes (Hyndman & Athanasopoulos), con tendencia
opcional (aditiva, amortiguada o no) y estacionalidad aditiva o multiplicativa.

    nivel      l_t = a*(y_t - s_{t-m})        + (1-a)*(l_{t-1} + phi*b_{t-1})   [aditivo]
    tendencia  b_t = b*(l_t - l_{t-1})        + (1-b)*phi*b_{t-1}
    estación   s_t = g*(y_t - l_{t-1} - phi*b_{t-1}) + (1-g)*s_{t-m}
    pronóstico yhat_{t|t-1} = l_{t-1} + phi*b_{t-1} + s_{t-m}

    (multiplicativo: y/s en el nivel, y/(l+phi*b) en la estación, yhat=(l+phi*b)*s)

Los parámetros (a, b, g, phi) se estiman minimizando la suma de cuadrados de los
errores de UN paso adelante sobre el bloque de entrenamiento. Los estados iniciales
se fijan con una heurística calculada SOLO con los primeros ciclos del entrenamiento.
Con `transform='log1p'` el modelo aditivo se ajusta sobre log(1+y) y las salidas se
devuelven en conteos (expm1, recortadas a >= 0).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

EPS = 1e-9


@dataclass
class HWConfig:
    trend: str | None = None      # None | 'add'
    damped: bool = False          # solo con trend='add'
    seasonal: str = "add"         # 'add' | 'mul'
    m: int = 24
    transform: str = "none"       # 'none' | 'log1p'

    @property
    def name(self) -> str:
        t = "sin_tend" if self.trend is None else ("tend_amort" if self.damped else "tend_lineal")
        tr = "_log1p" if self.transform == "log1p" else ""
        return f"HW_{t}_{'estac_mult' if self.seasonal == 'mul' else 'estac_adit'}{tr}"


def _initial_states(y: np.ndarray, cfg: HWConfig):
    """Heurística clásica: nivel/tendencia de los 2 primeros ciclos y estaciones de los primeros ciclos."""
    m = cfg.m
    if len(y) < 2 * m:
        raise ValueError(f"Se requieren al menos {2*m} observaciones para m={m}.")
    k = max(2, min(len(y) // m, 4))                       # ciclos usados para estacionalidad
    cycles = y[: k * m].reshape(k, m)
    cmeans = cycles.mean(axis=1, keepdims=True)
    if cfg.seasonal == "add":
        seas = (cycles - cmeans).mean(axis=0)
    else:
        seas = (cycles / np.maximum(cmeans, EPS)).mean(axis=0)
        seas = seas / max(seas.mean(), EPS)
    l0 = float(y[:m].mean())
    b0 = float((y[m:2 * m].mean() - y[:m].mean()) / m) if cfg.trend else 0.0
    return l0, b0, seas.astype(float)


def _run(y, a, b, g, phi, l0, b0, seas0, m, trend, seasonal):
    """Recursión de filtrado. Devuelve predicciones de un paso y los estados finales."""
    n = len(y)
    s = list(seas0)                      # s[t % m] guarda s_{t-m} antes de actualizar
    l, bt = l0, (b0 if trend else 0.0)
    ph = phi if trend else 0.0           # sin tendencia: b=0 y phi irrelevante
    yhat = np.empty(n)
    mult = seasonal == "mul"
    for t in range(n):
        base = l + ph * bt
        st = s[t % m]
        pred = base * st if mult else base + st
        yhat[t] = pred
        yt = y[t]
        if mult:
            l_new = a * (yt / max(st, EPS)) + (1 - a) * base
            s_new = g * (yt / max(base, EPS)) + (1 - g) * st
        else:
            l_new = a * (yt - st) + (1 - a) * base
            s_new = g * (yt - base) + (1 - g) * st
        if trend:
            bt = b * (l_new - l) + (1 - b) * ph * bt
        l = l_new
        s[t % m] = s_new
    seas_next = np.array([s[(n + i) % m] for i in range(m)])   # estaciones para t = n, n+1, ...
    return yhat, l, bt, seas_next


@dataclass
class HoltWinters:
    cfg: HWConfig = field(default_factory=HWConfig)
    params_: dict | None = None
    init_: tuple | None = None
    state_: tuple | None = None
    fitted_: np.ndarray | None = None
    sse_: float | None = None

    # ---------------- transformaciones ----------------
    def _fwd(self, y):
        y = np.asarray(y, float)
        return np.log1p(y) if self.cfg.transform == "log1p" else y

    def _inv(self, z):
        z = np.asarray(z, float)
        out = np.expm1(z) if self.cfg.transform == "log1p" else z
        return np.maximum(out, 0.0)

    # ---------------- estimación ----------------
    def fit(self, y):
        c = self.cfg
        z = self._fwd(y)
        if c.seasonal == "mul" and z.min() <= 0:
            raise ValueError("La estacionalidad multiplicativa requiere y > 0.")
        l0, b0, seas0 = _initial_states(z, c)
        self.init_ = (l0, b0, seas0)

        names = ["alpha", "gamma"] + (["beta"] if c.trend else []) + (["phi"] if (c.trend and c.damped) else [])
        bounds = {"alpha": (1e-4, 1 - 1e-4), "gamma": (1e-4, 1 - 1e-4),
                  "beta": (1e-4, 1 - 1e-4), "phi": (0.80, 0.995)}

        def unpack(x):
            d = dict(zip(names, x))
            return d["alpha"], d.get("beta", 0.0), d["gamma"], d.get("phi", 1.0)

        def sse(x):
            a, b, g, phi = unpack(x)
            yh, *_ = _run(z, a, b, g, phi, l0, b0, seas0, c.m, c.trend, c.seasonal)
            e = z - yh
            v = float(e @ e)
            return v if np.isfinite(v) else 1e30

        starts = [(0.3, 0.1, 0.1, 0.95), (0.1, 0.05, 0.3, 0.9), (0.6, 0.2, 0.05, 0.98)]
        best = None
        for st in starts:
            x0 = [{"alpha": st[0], "beta": st[1], "gamma": st[2], "phi": st[3]}[nm] for nm in names]
            r = minimize(sse, x0, method="L-BFGS-B", bounds=[bounds[n] for n in names])
            if best is None or r.fun < best.fun:
                best = r
        a, b, g, phi = unpack(best.x)
        self.params_ = {"alpha": a, "gamma": g}
        if c.trend:
            self.params_["beta"] = b
            self.params_["phi"] = phi if c.damped else 1.0
        yh, l, bt, sn = _run(z, a, b, g, self.params_.get("phi", 1.0), l0, b0, seas0, c.m, c.trend, c.seasonal)
        self.state_ = (l, bt, sn)
        self.fitted_ = self._inv(yh)
        self.sse_ = float(best.fun)
        return self

    # ---------------- uso con parámetros fijos ----------------
    def one_step(self, y_full):
        """Predicciones de 1 paso para TODA la serie con parámetros congelados.
        La predicción en t solo usa y_0..y_{t-1}: evaluación honesta de validación/prueba
        (los estados se actualizan con lo observado; los parámetros no se reestiman)."""
        p, (l0, b0, s0) = self.params_, self.init_
        yh, *_ = _run(self._fwd(y_full), p["alpha"], p.get("beta", 0.0), p["gamma"], p.get("phi", 1.0),
                      l0, b0, s0, self.cfg.m, self.cfg.trend, self.cfg.seasonal)
        return self._inv(yh)

    def forecast(self, h: int, y_full=None):
        """Pronóstico multi-paso desde el final de `y_full` (o del ajuste), sin datos futuros."""
        p, (l0, b0, s0) = self.params_, self.init_
        c = self.cfg
        if y_full is None:
            l, bt, sn = self.state_
        else:
            _, l, bt, sn = _run(self._fwd(y_full), p["alpha"], p.get("beta", 0.0), p["gamma"], p.get("phi", 1.0),
                                l0, b0, s0, c.m, c.trend, c.seasonal)
        phi = p.get("phi", 1.0) if c.trend else 0.0
        out = np.empty(h)
        for i in range(1, h + 1):
            damp = sum(phi ** j for j in range(1, i + 1)) if c.trend else 0.0
            base = l + damp * bt
            sv = sn[(i - 1) % c.m]
            out[i - 1] = base * sv if c.seasonal == "mul" else base + sv
        return self._inv(out)

    def components(self, y_full):
        """Nivel, tendencia y estacionalidad filtrados a lo largo de la serie (escala del modelo)."""
        p, (l0, b0, s0) = self.params_, self.init_
        c = self.cfg
        z = self._fwd(y_full)
        a, b, g, phi = p["alpha"], p.get("beta", 0.0), p["gamma"], p.get("phi", 1.0)
        ph = phi if c.trend else 0.0
        s = list(s0); l, bt = l0, (b0 if c.trend else 0.0)
        L, B, S = [], [], []
        for t in range(len(z)):
            base = l + ph * bt; st = s[t % c.m]
            if c.seasonal == "mul":
                ln = a * z[t] / max(st, EPS) + (1 - a) * base; sn_ = g * z[t] / max(base, EPS) + (1 - g) * st
            else:
                ln = a * (z[t] - st) + (1 - a) * base; sn_ = g * (z[t] - base) + (1 - g) * st
            if c.trend:
                bt = b * (ln - l) + (1 - b) * ph * bt
            l = ln; s[t % c.m] = sn_
            L.append(l); B.append(bt); S.append(sn_)
        return np.array(L), np.array(B), np.array(S)


def hw_candidates(m: int = 24, y_min: float = 1.0):
    """Rejilla fija y pequeña de candidatos (se elige por MAE de validación)."""
    cands = []
    for trend, damped in [(None, False), ("add", True)]:
        for seasonal in ["add", "mul"]:
            if seasonal == "mul" and y_min <= 0:
                continue
            cands.append(HWConfig(trend, damped, seasonal, m))
    cands.append(HWConfig(None, False, "add", m, "log1p"))
    cands.append(HWConfig("add", True, "add", m, "log1p"))
    return cands
