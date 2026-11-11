"""Genera notebooks/Laboratorio_3.ipynb (JSON nbformat 4.5, sin dependencias). Ejecutarlo en Jupyter: Run All."""
from __future__ import annotations
import json, uuid
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def md(t): return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}

cells = [
md("""# Laboratorio 3 · PaySim adaptado: regresión lineal y Holt-Winters
**Estadística para Analítica de Datos · UPTC · 2026-II · Prof. Duván Cataño** — David Nuncira, Juan Valderrama, Daniel Carrillo

Proyecto de modelación sobre la serie horaria de transacciones del Laboratorio 2. Ejecutar con *Run All* (la serie se lee de `results/hourly_series.csv`, igual que en el Laboratorio 2)."""),
code("""from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), Path.cwd().parent, Path('/content/laboratorio_3_paysim')] if (p/'src'/'pipeline.py').exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display, Markdown
except ImportError:
    display = print; Markdown = lambda s: s
from src.data import load_series, TARGET
from src.pipeline import run_all
from src.modeling import make_features, decompose, acf
from src.holtwinters import HoltWinters, HWConfig
from src.narrative import findings, hypotheses_text, diag_test
pd.options.display.float_format = '{:,.4f}'.format
plt.rcParams.update({'figure.figsize': (11, 3.6), 'axes.spines.top': False, 'axes.spines.right': False})
d = load_series(); o = run_all(d); sp = o['splits']
print(f"{len(d)} horas, {int(d[TARGET].sum()):,} transacciones | partición:", sp)"""),
md("""## 1. Introducción y objetivo
Se pronostica `transacciones_hora` (conteo por hora simulada). Es un problema de **regresión** (la guía también lista regresión logística, que no aplica: no hay etiqueta de clase en el dataset adaptado). Modelos: **regresión lineal** (rezagos y variables temporales), cinco **baselines** y **Holt-Winters** (suavizamiento exponencial triple, periodo estacional 24 h)."""),
md("## 2. Descripción de variables"),
code("""display(pd.DataFrame({
    'variable': ['step', 'transacciones_hora', 'monto_total / promedio / mediano', 'fase_24', 'lag_1, lag_24', 'step + sin/cos(2πt/24, 2πt/12)'],
    'rol': ['índice temporal', 'OBJETIVO', 'NO predictor (se conoce al cierre de la hora: fuga)', 'perfil intradiario (step % 24)', 'predictores (conocidos en t-1)', 'predictores deterministas']}))
d.head()"""),
md("## 3. Limpieza y preprocesamiento\nValidación de finitud, no negatividad, unicidad y continuidad de `step` (en `load_series`). No se imputa ni se recortan picos. Partición cronológica sin barajar."),
code("""x = make_features(d)
assert d.step.diff().dropna().eq(1).all() and d[TARGET].notna().all()
display(o['splits_table']); x.head()"""),
md("## 4. Análisis exploratorio (EDA)"),
code("""display(o['descriptive']); display(o['regimes'])
fig, ax = plt.subplots(); ax.plot(d.step, d[TARGET], lw=1, color='#1f3b4d')
for v in (sp['train_end'], sp['val_end']): ax.axvline(v, ls='--', color='crimson')
ax.set(title='Transacciones por hora (líneas: fin de train y de validación)', xlabel='step'); plt.show()"""),
code("""fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
axs[0].hist(d[TARGET], bins=40, color='#1f3b4d'); axs[0].set_title('Distribución')
prof = d[d.step <= sp['train_end']].groupby('fase_24')[TARGET].mean(); axs[1].plot(prof, marker='o', color='#e4572e'); axs[1].set_title('Perfil por fase (train)')
a = o['acf']; axs[2].bar(a.lag[1:73], a.acf[1:73], color='#2a9d8f'); axs[2].set_title('ACF'); plt.show()
print(o['acf_resumen'])"""),
code("""dc = decompose(d[TARGET].to_numpy(), 24, 'additive')   # descomposición clásica aditiva
fig, axs = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
for ax, c in zip(axs, ['tendencia', 'estacionalidad', 'residuo']): ax.plot(d.step, dc[c], lw=1); ax.set_title(c)
plt.tight_layout(); plt.show()"""),
md("## 5. Pruebas de hipótesis e interpretación\nα = 0,05, solo *train*. H1/H2: Wald con covarianza HAC (Newey-West, 24 rezagos); H3: Kruskal-Wallis entre las 24 fases; corrección de Holm. Ljung-Box de residuos de Holt-Winters como diagnóstico."),
code("""display(o['hypotheses']); display(o['coef_hac']); display(o['ljungbox_hw'])
for t in hypotheses_text(o): display(Markdown('- ' + t))"""),
md("""## 6. Construcción de pipelines
**Lineal:** columnas → `StandardScaler` (ajustado solo con *train*) → `LinearRegression`; salida `max(0, ŷ)`.

**Holt-Winters** (ver `src/holtwinters.py`), con periodo m = 24:

$$\\hat y_{t|t-1}=\\ell_{t-1}+\\phi b_{t-1}+s_{t-m},\\quad \\ell_t=\\alpha(y_t-s_{t-m})+(1-\\alpha)(\\ell_{t-1}+\\phi b_{t-1})$$
$$b_t=\\beta(\\ell_t-\\ell_{t-1})+(1-\\beta)\\phi b_{t-1},\\quad s_t=\\gamma(y_t-\\ell_{t-1}-\\phi b_{t-1})+(1-\\gamma)s_{t-m}$$

La versión multiplicativa usa $(\\ell+\\phi b)\\,s$ y cocientes. Los parámetros se estiman minimizando el SSE de un paso en *train*; los estados iniciales salen de los primeros ciclos de *train*."""),
code("""from sklearn.pipeline import Pipeline
from src.modeling import build_pipeline, LAGS, CALENDAR
print(build_pipeline(LAGS + CALENDAR))
# Verificación opcional contra statsmodels (solo si está instalado): mismos parámetros y estados iniciales
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    cfg = HWConfig(None, False, 'add', 24); y = d[TARGET].to_numpy(float)[:sp['train_end']]
    mine = HoltWinters(cfg).fit(y); l0, b0, s0 = mine.init_; p = mine.params_
    sm = ExponentialSmoothing(y, trend=None, seasonal='add', seasonal_periods=24, initialization_method='known',
                              initial_level=l0, initial_seasonal=s0).fit(smoothing_level=p['alpha'], smoothing_seasonal=p['gamma'], optimized=False)
    print('máx |diferencia| vs statsmodels:', float(np.abs(sm.fittedvalues - mine.fitted_).max()))
except Exception as e:
    print('Verificación con statsmodels omitida:', type(e).__name__)"""),
md("## 7. Entrenamiento de modelos\nValidación temporal expansiva (3 bloques de 72 h), comparación en validación y selección por MAE **sin usar test**. Candidatos Holt-Winters: tendencia (ninguna / amortiguada) × estacionalidad (aditiva / multiplicativa) y versiones aditivas sobre log(1+y)."),
code("""display(o['hw_grid_validation']); display(o['validation'][['modelo','familia','MAE','RMSE','MAPE_pct','R2']])
display(o['cv'].groupby('modelo')[['MAE','RMSE','R2']].mean())
print(o['selected'])"""),
code("""hw = o['hw_final']; L, B, S = hw.components(d[TARGET].to_numpy())
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(11, 5)); axs[0].plot(d.step, L, color='#e4572e'); axs[0].set_title('Nivel (Holt-Winters)')
axs[1].plot(d.step, S, color='#2a9d8f'); axs[1].set_title('Componente estacional'); plt.show()
display(o['hw_params'])"""),
md("## 8. Evaluación del modelo\nPrueba: modelos reajustados con *train+validación*, evaluados **una sola vez**. Protocolo principal: pronóstico de 1 hora (parámetros congelados, estados actualizados con lo observado). Se reporta además el horizonte completo con origen fijo."),
code("""cols = ['modelo','familia','MAE','RMSE','MAPE_pct','WAPE_pct','R2']
display(o['test'][cols]); display(o['test_origen_fijo'][cols])
display(diag_test(o))   # sesgo y correlación en prueba"""),
code("""p = o['predictions']; te = p[p.bloque == 'test']; sel = o['selected']
fig, ax = plt.subplots(); ax.plot(te.step, te.real, color='#1f3b4d', lw=1.6, label='Real')
ax.plot(te.step, te[sel['holt_winters']], color='#e4572e', label='Holt-Winters (1 h)')
ax.plot(te.step, te[sel['holt_winters'] + '_origen_fijo'], color='#e4572e', ls=':', label='Holt-Winters origen fijo')
ax.plot(te.step, te[sel['lineal']], color='#2a9d8f', label='Lineal (1 h)'); ax.legend(); ax.set_title('Prueba'); plt.show()
fc = o['forecast_72']; h = d.tail(168); fig, ax = plt.subplots(); ax.plot(h.step, h[TARGET], color='#1f3b4d'); ax.plot(fc.step, fc.pronostico_HW, color='#e4572e')
ax.set_title('Pronóstico Holt-Winters a 72 h (ajustado con toda la serie)'); plt.show()"""),
md("## 9. Conclusiones\n(Generadas a partir de los resultados calculados arriba.)"),
code("""for t in findings(o)['texto'] + findings(o)['alertas']: display(Markdown('- ' + t))
display(Markdown('- Limitaciones: serie simulada y corta, cambio de nivel entre bloques, conteos pequeños que inflan MAPE, un solo bloque de prueba y sin intervalos validados.'))"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "Laboratorio_3.ipynb"
out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(out)
