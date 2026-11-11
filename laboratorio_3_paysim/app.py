"""Dashboard interactivo · Laboratorio 3 PaySim (regresión lineal + Holt-Winters).
Ejecutar:  python -m streamlit run app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data import TARGET, find_series_path, load_series
from src.holtwinters import HoltWinters, HWConfig
from src.modeling import CALENDAR, LAGS, acf, decompose, metrics, predict_linear
from src.narrative import f as fmt, findings, hypotheses_text, diag_test
from src.pipeline import recursive_linear, run_all

# ------------------------------------------------------------------ estilo
st.set_page_config(page_title="PaySim · Pronóstico Holt-Winters", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
PAL = {"real": "#1f3b4d", "hw": "#e4572e", "lin": "#2a9d8f", "base": "#8d99ae", "acc": "#f4a261"}
FAM = {"holt_winters": PAL["hw"], "regresion_lineal": PAL["lin"], "baseline": PAL["base"]}
BLK = {"train": "rgba(42,157,143,0.07)", "validacion": "rgba(244,162,97,0.16)", "test": "rgba(228,87,46,0.12)"}
st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.hero {background: linear-gradient(120deg,#1f3b4d 0%,#2a9d8f 60%,#e4572e 140%); color:#fff; padding:1.1rem 1.4rem;
       border-radius:14px; margin-bottom:1rem;}
.hero h1 {color:#fff; margin:0; font-size:1.7rem;} .hero p {margin:.2rem 0 0 0; opacity:.9;}
[data-testid="stMetric"] {background:#f3f6f9; border-left:5px solid #e4572e; padding:.7rem .9rem; border-radius:10px;}
[data-testid="stMetricLabel"] p {font-weight:600; color:#5b6b7a;}
.stTabs [data-baseweb="tab"] {font-weight:600;}
</style>""", unsafe_allow_html=True)


def style(fig, h=420, title=None):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      title=title, hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
                      font=dict(family="Inter, Segoe UI, sans-serif", color=PAL["real"]))
    return fig


def show_df(df: pd.DataFrame, dec=2):
    num = df.select_dtypes("number").columns
    st.dataframe(df.style.format({c: f"{{:,.{dec}f}}" for c in num}), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ datos y modelos (con caché)
@st.cache_data(show_spinner="Cargando serie horaria…")
def get_data():
    return load_series()


@st.cache_resource(show_spinner="Entrenando regresión lineal, baselines y Holt-Winters (primera vez ~10 s)…")
def get_results():
    return run_all(get_data())


@st.cache_data(show_spinner="Ajustando Holt-Winters…")
def hw_fit(y, trend, damped, seasonal, transform, m, n_fit, h):
    cfg = HWConfig(trend, damped, seasonal, m, transform)
    mdl = HoltWinters(cfg).fit(y[:n_fit])
    one = mdl.one_step(y)
    multi = mdl.forecast(len(y) - n_fit, y[:n_fit]) if n_fit < len(y) else np.array([])
    lvl, _, seas = mdl.components(y)
    return {"params": mdl.params_, "one": one, "multi": multi, "level": lvl, "seas": seas,
            "future": mdl.forecast(h, y), "fitted": mdl.fitted_}


if find_series_path() is None:
    st.error("No se encontró la serie horaria. Copia `results/hourly_series.csv` del Laboratorio 2 en la carpeta "
             "`results/` (o `data/`) de este proyecto y recarga la página.")
    st.stop()

d, o = get_data(), get_results()
sp, sel, y = o["splits"], o["selected"], d[TARGET].to_numpy(float)
steps = d.step.to_numpy(); N = len(d); i_tr = int((steps <= sp["train_end"]).sum()); i_va = int((steps <= sp["val_end"]).sum())
d = d.assign(bloque=np.where(d.step <= sp["train_end"], "train", np.where(d.step <= sp["val_end"], "validacion", "test")))

# ------------------------------------------------------------------ sidebar (filtros)
sb = st.sidebar
sb.markdown("## 🎛️ Filtros")
r_steps = sb.slider("Rango de horas (step)", int(d.step.min()), int(d.step.max()), (int(d.step.min()), int(d.step.max())))
blocks = sb.multiselect("Bloques", ["train", "validacion", "test"], default=["train", "validacion", "test"])
r_fase = sb.slider("Fase del ciclo diario (step % 24)", 0, 23, (0, 23))
win = sb.slider("Media móvil (horas)", 1, 48, 6)
logy = sb.toggle("Escala logarítmica en Y", value=False)
sb.markdown("## 🔮 Holt-Winters")
use_sel = sb.checkbox(f"Usar el elegido en validación ({sel['holt_winters']})", value=True)
if use_sel:
    hcfg = o["hw_final"].cfg
else:
    t_lbl = sb.selectbox("Tendencia", ["ninguna", "amortiguada"])
    s_lbl = sb.selectbox("Estacionalidad", ["aditiva", "multiplicativa"])
    x_lbl = sb.selectbox("Transformación", ["ninguna", "log1p"])
    m_hw = sb.selectbox("Periodo estacional (h)", [12, 24, 48], index=1)
    hcfg = HWConfig("add" if t_lbl == "amortiguada" else None, t_lbl == "amortiguada",
                    "mul" if s_lbl == "multiplicativa" else "add", m_hw, "log1p" if x_lbl == "log1p" else "none")
horizon = sb.slider("Horizonte de pronóstico (h)", 1, 72, 24)
sb.markdown("## ⚖️ Comparación")
test_models = list(o["test"].modelo)
default_models = [m for m in dict.fromkeys([sel["holt_winters"], sel["lineal"], "persistencia_1", "estacional_24", "media_estacional_7"]) if m in test_models]
cmp_models = sb.multiselect("Modelos", test_models, default=default_models)
metric = sb.radio("Métrica", ["MAE", "RMSE", "MAPE_pct", "WAPE_pct", "R2"], horizontal=True)

mask = d.step.between(*r_steps) & d.bloque.isin(blocks) & d.fase_24.between(*r_fase)
df = d[mask]
if df.empty:
    st.warning("Los filtros no dejan ninguna hora. Ajusta el rango, los bloques o la fase.")
    st.stop()

st.markdown("""<div class="hero"><h1>📈 PaySim · Pronóstico del volumen horario de transacciones</h1>
<p>Laboratorio 3 · Regresión lineal y Holt-Winters · Estadística para Analítica de Datos · UPTC 2026-II</p></div>""",
            unsafe_allow_html=True)

tabs = st.tabs(["📊 Resumen", "🔍 EDA", "🧩 Descomposición", "🔮 Holt-Winters", "⚖️ Comparación", "🎯 Pronóstico", "📚 Proyecto"])

# ================================================================== RESUMEN
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Horas filtradas", fmt(len(df), 0), help="Observaciones que cumplen los filtros")
    c[1].metric("Transacciones", fmt(df[TARGET].sum(), 0))
    c[2].metric("Media horaria", fmt(df[TARGET].mean(), 0), delta=f"mediana {fmt(df[TARGET].median(), 0)}", delta_color="off")
    c[3].metric("Máximo horario", fmt(df[TARGET].max(), 0), delta=f"step {int(df.loc[df[TARGET].idxmax(), 'step'])}", delta_color="off")
    F = findings(o)
    c[4].metric("MAE test · Holt-Winters", fmt(F["thw"].MAE, 1), delta=f"{fmt(F['thw'].MAE - F['tlin'].MAE, 1)} vs lineal", delta_color="inverse")

    fig = go.Figure()
    mode = "lines" if len(df) == (r_steps[1] - r_steps[0] + 1) else "lines+markers"
    fig.add_trace(go.Scatter(x=df.step, y=df[TARGET], mode=mode, name="Transacciones", line=dict(color=PAL["real"], width=1.2),
                             marker=dict(size=4)))
    if win > 1:
        fig.add_trace(go.Scatter(x=df.step, y=df[TARGET].rolling(win, min_periods=1).mean(), name=f"Media móvil {win} h",
                                 line=dict(color=PAL["hw"], width=2.4)))
    for name, a, b in [("train", d.step.min(), sp["train_end"]), ("validación", sp["train_end"], sp["val_end"]), ("test", sp["val_end"], d.step.max())]:
        fig.add_vrect(x0=a, x1=b, fillcolor=BLK["validacion" if name == "validación" else name], line_width=0, layer="below",
                      annotation_text=name, annotation_position="top left")
    fig.update_xaxes(rangeslider_visible=True, title="step (hora simulada)")
    fig.update_yaxes(type="log" if logy else "linear", title="transacciones")
    st.plotly_chart(style(fig, 480, "Serie horaria con partición cronológica"), use_container_width=True)

    st.markdown("#### Hallazgos clave (calculados con los datos)")
    for t in F["texto"]:
        st.markdown("- " + t)
    for t in F["alertas"]:
        st.warning(t)

# ================================================================== EDA
with tabs[1]:
    a1, a2 = st.columns(2)
    with a1:
        nb = st.slider("Barras del histograma", 10, 100, 40)
        fig = px.histogram(df, x=TARGET, nbins=nb, color_discrete_sequence=[PAL["real"]], log_y=logy)
        st.plotly_chart(style(fig, 380, "Distribución de transacciones por hora"), use_container_width=True)
    with a2:
        fig = px.box(df, x="fase_24", y=TARGET, color_discrete_sequence=[PAL["lin"]], log_y=logy, points="outliers")
        fig.update_xaxes(title="step % 24")
        st.plotly_chart(style(fig, 380 + 52, "Distribución por fase del ciclo"), use_container_width=True)

    piv = df.pivot_table(index="ciclo", columns="fase_24", values=TARGET, aggfunc="mean")
    z = np.log10(piv.clip(lower=1)) if logy else piv
    fig = go.Figure(go.Heatmap(z=z.values, x=piv.columns, y=piv.index, colorscale="Turbo", customdata=piv.values,
                               hovertemplate="ciclo %{y} · fase %{x}<br>%{customdata:,.0f} transacciones<extra></extra>",
                               colorbar=dict(title="log10" if logy else "trans.")))
    fig.update_layout(xaxis_title="fase del ciclo (step % 24)", yaxis_title="ciclo de 24 h")
    st.plotly_chart(style(fig, 420, "Mapa de calor: ciclo × hora (se ve el cambio de régimen)"), use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        seg = d[d.step.between(*r_steps)][TARGET].to_numpy(float)
        nl = int(min(72, max(len(seg) - 3, 2)))
        ac = acf(seg, nl); band = 1.96 / np.sqrt(len(seg))
        fig = go.Figure(go.Bar(x=np.arange(1, nl + 1), y=ac[1:], marker_color=PAL["acc"]))
        fig.add_hline(y=band, line_dash="dash", line_color="gray"); fig.add_hline(y=-band, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_title="rezago (h)", yaxis_title="ACF")
        st.plotly_chart(style(fig, 360, "Autocorrelación (rango de horas elegido)"), use_container_width=True)
    with b2:
        lag = st.radio("Rezago", [1, 24], horizontal=True)
        sc = pd.DataFrame({"x": d[TARGET].shift(lag), "y": d[TARGET], "step": d.step}).dropna()
        sc = sc[sc.step.between(*r_steps)]
        fig = px.scatter(sc, x="x", y="y", color="step", color_continuous_scale="Viridis", opacity=.7,
                         labels={"x": f"y(t-{lag})", "y": "y(t)"}, log_x=logy, log_y=logy)
        st.plotly_chart(style(fig, 330, f"Relación con el rezago de {lag} h"), use_container_width=True)
    with st.expander("Descriptivas por bloques de 168 horas"):
        show_df(o["regimes"], 1)

# ================================================================== DESCOMPOSICIÓN
with tabs[2]:
    kind = st.radio("Tipo de descomposición clásica (periodo 24 h)", ["aditiva", "multiplicativa"], horizontal=True)
    seg = d[d.step.between(*r_steps)]
    if len(seg) < 72:
        st.info("Elige un rango de al menos 72 horas para descomponer.")
    elif kind == "multiplicativa" and (seg[TARGET] <= 0).any():
        st.warning("La descomposición multiplicativa requiere valores positivos.")
    else:
        dc = decompose(seg[TARGET].to_numpy(), 24, "additive" if kind == "aditiva" else "multiplicative")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.04,
                            subplot_titles=["Observado", "Tendencia (media móvil 2×24)", "Estacionalidad", "Residuo"])
        for i, (col, colr) in enumerate(zip(dc.columns, [PAL["real"], PAL["hw"], PAL["lin"], PAL["base"]]), 1):
            fig.add_trace(go.Scatter(x=seg.step, y=dc[col], line=dict(color=colr, width=1.3), showlegend=False), row=i, col=1)
        fig.update_layout(height=720)
        st.plotly_chart(style(fig, 720), use_container_width=True)
        fuerza = max(0.0, 1 - np.nanvar(dc.residuo) / np.nanvar(dc.residuo + dc.estacionalidad)) if kind == "aditiva" else np.nan
        if kind == "aditiva":
            st.caption(f"Fuerza de la estacionalidad (Hyndman): {fuerza:.2f} (0 = nula, 1 = total).")

# ================================================================== HOLT-WINTERS
with tabs[3]:
    st.markdown(f"**Configuración activa:** `{hcfg.name}` (m = {hcfg.m}). Parámetros estimados minimizando el SSE de un paso adelante.")
    scope = st.radio("Ajustar con", ["Entrenamiento", "Entrenamiento + validación", "Toda la serie"], index=1, horizontal=True)
    n_fit = {"Entrenamiento": i_tr, "Entrenamiento + validación": i_va, "Toda la serie": N}[scope]
    try:
        R = hw_fit(y, hcfg.trend, hcfg.damped, hcfg.seasonal, hcfg.transform, hcfg.m, n_fit, horizon)
    except ValueError as e:
        st.error(f"No se pudo ajustar este modelo: {e}")
        R = None
    if R is not None:
        pc = st.columns(len(R["params"]) + 1)
        for k, (nm, v) in enumerate(R["params"].items()):
            pc[k].metric({"alpha": "α (nivel)", "beta": "β (tendencia)", "gamma": "γ (estación)", "phi": "φ (amortiguación)"}[nm], f"{v:.4f}")
        pc[-1].metric("Observaciones de ajuste", fmt(n_fit, 0))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=y, name="Real", line=dict(color=PAL["real"], width=1.6)))
        fig.add_trace(go.Scatter(x=steps[:n_fit], y=R["one"][:n_fit], name="Ajuste en muestra (1 h)", line=dict(color=PAL["acc"], width=1)))
        if n_fit < N:
            fig.add_trace(go.Scatter(x=steps[n_fit:], y=R["one"][n_fit:], name="Pronóstico 1 h (params. fijos)", line=dict(color=PAL["hw"], width=1.8)))
            fig.add_trace(go.Scatter(x=steps[n_fit:], y=R["multi"], name="Pronóstico origen fijo", line=dict(color=PAL["lin"], width=1.8, dash="dot")))
            fig.add_vline(x=float(steps[n_fit - 1]), line_dash="dash", line_color="gray")
        else:
            fut_x = np.arange(steps[-1] + 1, steps[-1] + 1 + horizon)
            fig.add_trace(go.Scatter(x=fut_x, y=R["future"], name=f"Pronóstico {horizon} h", line=dict(color=PAL["hw"], width=2.4)))
        fig.update_xaxes(rangeslider_visible=True, title="step"); fig.update_yaxes(type="log" if logy else "linear", title="transacciones")
        st.plotly_chart(style(fig, 460, "Holt-Winters: real, ajuste y pronósticos"), use_container_width=True)

        if n_fit < N:
            mm = st.columns(2)
            m1, m2 = metrics(y[n_fit:], R["one"][n_fit:]), metrics(y[n_fit:], R["multi"])
            with mm[0]:
                st.markdown("**Un paso adelante (después del ajuste)**")
                show_df(pd.DataFrame([m1]).drop(columns="n"), 3)
            with mm[1]:
                st.markdown("**Origen fijo (horizonte completo, sin ver datos nuevos)**")
                show_df(pd.DataFrame([m2]).drop(columns="n"), 3)
        else:
            st.info("Con «Toda la serie» no queda bloque fuera de muestra; elige otro ajuste para evaluar.")

        lc1, lc2 = st.columns(2)
        with lc1:
            fig = go.Figure(go.Scatter(x=steps, y=R["level"], line=dict(color=PAL["hw"])))
            st.plotly_chart(style(fig, 300, "Nivel filtrado" + (" (escala log1p)" if hcfg.transform == "log1p" else "")), use_container_width=True)
        with lc2:
            fig = go.Figure(go.Scatter(x=steps, y=R["seas"], line=dict(color=PAL["lin"])))
            st.plotly_chart(style(fig, 300, "Componente estacional filtrado"), use_container_width=True)

    st.markdown("#### Rejilla de candidatos evaluada en validación")
    g = o["hw_grid_validation"]
    fig = px.bar(g.sort_values("MAE", ascending=False), x="MAE", y="modelo", orientation="h", color_discrete_sequence=[PAL["hw"]])
    st.plotly_chart(style(fig, 330), use_container_width=True)
    show_df(g, 3)

# ================================================================== COMPARACIÓN
with tabs[4]:
    tt = o["test"][o["test"].modelo.isin(cmp_models)].copy()
    vt = o["validation"][o["validation"].modelo.isin(cmp_models)].assign(bloque="validacion")
    tt2 = tt.assign(bloque="test")
    both = pd.concat([vt, tt2])
    fig = px.bar(both, x="modelo", y=metric, color="bloque", barmode="group",
                 color_discrete_map={"validacion": PAL["acc"], "test": PAL["hw"]}, text_auto=".3s")
    st.plotly_chart(style(fig, 420, f"{metric} por modelo: validación vs prueba (1 h)"), use_container_width=True)
    show_df(tt[["modelo", "familia", "MAE", "RMSE", "MAPE_pct", "WAPE_pct", "R2"]], 3)
    st.caption("Los modelos se eligen con validación; la prueba se evalúa una sola vez con los modelos reajustados con train+validación. "
               "Diferencias pequeñas en un único bloque de prueba no son concluyentes.")

    st.markdown("#### Diagnóstico de un modelo")
    d1, d2, d3 = st.columns(3)
    mdl = d1.selectbox("Modelo", cmp_models or test_models)
    blk = d2.selectbox("Bloque", ["test", "validacion", "train"])
    pr = o["predictions"]; pr = pr[(pr.bloque == blk) & pr[mdl].notna()]
    err = pr["real"] - pr[mdl]
    d3.metric(f"MAE en {blk}", fmt(err.abs().mean(), 1))
    e1, e2 = st.columns(2)
    with e1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr.step, y=pr.real, name="Real", line=dict(color=PAL["real"], width=1.6)))
        fig.add_trace(go.Scatter(x=pr.step, y=pr[mdl], name=mdl, line=dict(color=PAL["hw"], width=1.6)))
        fig.update_yaxes(type="log" if logy else "linear")
        st.plotly_chart(style(fig, 360, "Real vs pronóstico"), use_container_width=True)
    with e2:
        mx = float(max(pr.real.max(), pr[mdl].max()))
        fig = go.Figure(go.Scatter(x=pr.real, y=pr[mdl], mode="markers", marker=dict(color=PAL["lin"], opacity=.65, size=6), name="horas"))
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="gray", dash="dash"), name="ideal"))
        fig.update_layout(xaxis_title="real", yaxis_title="pronóstico", hovermode="closest")
        st.plotly_chart(style(fig, 360, "Calibración (puntos sobre la diagonal = acierto)"), use_container_width=True)
    e3, e4 = st.columns(2)
    with e3:
        fig = px.histogram(err, nbins=40, color_discrete_sequence=[PAL["acc"]], labels={"value": "error (real - pronóstico)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 330, "Distribución del error"), use_container_width=True)
    with e4:
        by = pr.assign(fase=pr.step % 24, ae=err.abs()).groupby("fase").ae.mean().reset_index()
        fig = px.bar(by, x="fase", y="ae", color_discrete_sequence=[PAL["hw"]], labels={"ae": "MAE", "fase": "step % 24"})
        st.plotly_chart(style(fig, 330, "Error absoluto medio por fase del ciclo"), use_container_width=True)
    with st.expander("Sesgo y correlación en prueba (complementa al MAE)"):
        show_df(diag_test(o), 3)
    with st.expander("Validación cruzada temporal expansiva (3 bloques de 72 h en entrenamiento)"):
        show_df(o["cv"], 2)

# ================================================================== PRONÓSTICO
with tabs[5]:
    st.markdown(f"Pronóstico a **{horizon} h** desde el final de la serie (step {int(steps[-1])}) con `{hcfg.name}` ajustado con toda la serie.")
    try:
        RF = hw_fit(y, hcfg.trend, hcfg.damped, hcfg.seasonal, hcfg.transform, hcfg.m, N, horizon)
    except ValueError as e:
        st.error(f"No se pudo ajustar este modelo: {e}"); RF = None
    if RF is not None:
        lin_name = sel["lineal"]; lin_cols = LAGS if lin_name == "lineal_lags" else LAGS + CALENDAR
        rec = recursive_linear(o["models"]["lin_final"][lin_name], lin_cols, y, horizon, int(steps[-1]) + 1)
        fx = np.arange(steps[-1] + 1, steps[-1] + 1 + horizon)
        hist = d.tail(168)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.step, y=hist[TARGET], name="Observado (últimas 168 h)", line=dict(color=PAL["real"], width=1.6)))
        fig.add_trace(go.Scatter(x=fx, y=RF["future"], name="Holt-Winters", line=dict(color=PAL["hw"], width=2.6)))
        fig.add_trace(go.Scatter(x=fx, y=rec, name=f"{lin_name} (recursivo)", line=dict(color=PAL["lin"], width=2, dash="dot")))
        fig.update_yaxes(type="log" if logy else "linear", title="transacciones")
        st.plotly_chart(style(fig, 440, "Pronóstico puntual (sin intervalos: no se validó su cobertura)"), use_container_width=True)
        out = pd.DataFrame({"step": fx, "fase_24": fx % 24, "pronostico_holt_winters": RF["future"], f"pronostico_{lin_name}": rec})
        show_df(out, 1)
        st.download_button("⬇️ Descargar pronóstico (CSV)", out.to_csv(index=False).encode("utf-8"), "pronostico_holt_winters.csv", "text/csv")
        st.caption("El pronóstico recursivo acumula error con el horizonte; la comparación entre los dos modelos a largo plazo es solo ilustrativa.")

    st.markdown("---\n#### Predicción manual de la próxima hora (regresión lineal `lineal_lags`)")
    q1, q2, q3 = st.columns(3)
    v1 = q1.number_input("y(t-1): transacciones de la hora anterior", min_value=0.0, value=float(y[-1]), step=1.0)
    v24 = q2.number_input("y(t-24): transacciones de hace 24 h", min_value=0.0, value=float(y[-24]), step=1.0)
    manual = float(predict_linear(o["models"]["lin_final"]["lineal_lags"], pd.DataFrame([{"lag_1": v1, "lag_24": v24}]), LAGS)[0])
    q3.metric("Pronóstico lineal para t", fmt(manual, 0))
    coef = pd.DataFrame({"variable": ["intercepto"] + LAGS})
    lm = o["models"]["lin_final"]["lineal_lags"].named_steps["lm"]; scl = o["models"]["lin_final"]["lineal_lags"].named_steps["prep"].named_transformers_["sc"]
    coef["coeficiente"] = [float(lm.intercept_ - (lm.coef_ / scl.scale_ * scl.mean_).sum())] + list(lm.coef_ / scl.scale_)
    with st.expander("Ecuación del modelo en unidades originales"):
        show_df(coef, 4)

# ================================================================== PROYECTO
with tabs[6]:
    st.markdown(f"""
### Descripción del proyecto
Continúa el **PaySim adaptado** de los Laboratorios 1 y 2: {fmt(len(d), 0)} horas simuladas y {fmt(d[TARGET].sum(), 0)} transacciones agregadas.
**Objetivo:** pronosticar `transacciones_hora` (conteo). Se usa la rama de **regresión** de la guía; la regresión logística no aplica porque no hay etiqueta de clase.

**Metodología**
1. EDA (distribución, perfil intradiario, ACF, descomposición, cambio de régimen).
2. Pruebas de hipótesis (HAC/Newey-West + Holm; Kruskal-Wallis entre fases).
3. Pipelines: `StandardScaler → LinearRegression` con rezagos (y variante temporal), cinco baselines y **Holt-Winters** (m = 24).
4. Partición cronológica 60/20/20 (train 1-{sp['train_end']}, validación hasta {sp['val_end']}, test hasta {sp['last']}); selección solo con validación.
5. Evaluación con MAE, RMSE, MAPE y R² (más WAPE), a una hora y con origen fijo.

**Límites:** serie simulada y corta, cambio de nivel entre bloques, conteos pequeños que inflan MAPE, un único bloque de prueba, sin intervalos validados.
""")
    st.markdown("#### Pruebas de hipótesis (α = 0,05, solo entrenamiento)")
    show_df(o["hypotheses"], 4)
    for t in hypotheses_text(o):
        st.markdown("- " + t)
    with st.expander("Datos filtrados"):
        show_df(df[["step", TARGET, "fase_24", "ciclo", "bloque"]], 0)
        st.download_button("⬇️ Descargar datos filtrados (CSV)", df.to_csv(index=False).encode("utf-8"), "datos_filtrados.csv", "text/csv")
    with st.expander("Predicciones guardadas de todos los modelos"):
        st.download_button("⬇️ Descargar predicciones (CSV)", o["predictions"].to_csv(index=False).encode("utf-8"), "predicciones.csv", "text/csv")
