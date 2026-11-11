"""Texto interpretativo generado a partir de los resultados reales (nada de cifras escritas a mano)."""
from __future__ import annotations
import pandas as pd


def f(v, dec=2):
    s = f"{v:,.{dec}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")     # formato es-CO: 1.234,56


def p_fmt(p):
    return "< 0,0001" if p < 1e-4 else f(p, 4)


def md_table(df: pd.DataFrame, dec=2, pcols=()) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in pcols and isinstance(v, (int, float)):
                cells.append(p_fmt(v))
            elif isinstance(v, (bool,)):
                cells.append("Sí" if v else "No")
            elif isinstance(v, (int,)) or (hasattr(v, "dtype") and "int" in str(v.dtype)):
                cells.append(str(int(v)))
            elif isinstance(v, float) and v == v and float(v).is_integer():
                cells.append(f(v, 0))
            elif isinstance(v, float):
                cells.append(f(v, dec))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _row(df, model):
    return df.set_index("modelo").loc[model]


def diag_test(o: dict) -> pd.DataFrame:
    """Sesgo medio (real - pronóstico) y correlación con lo real en prueba (1 h): complementa al MAE."""
    pr = o["predictions"]; pr = pr[pr.bloque == "test"]
    rows = []
    for m in o["test"].modelo:
        if m in pr.columns and pr[m].notna().all() and pr[m].std() > 0:
            rows.append({"modelo": m, "pronostico_medio": float(pr[m].mean()), "sesgo_medio": float((pr.real - pr[m]).mean()),
                         "correlacion": float(pr.real.corr(pr[m]))})
    return pd.DataFrame(rows)


def findings(o: dict) -> dict:
    sel, te, fx = o["selected"], o["test"], o["test_origen_fijo"]
    hw, lin = sel["holt_winters"], sel["lineal"]
    thw, tlin = _row(te, hw), _row(te, lin)
    pers, s24 = _row(te, "persistencia_1"), _row(te, "estacional_24")
    best = te.iloc[0]
    rank_hw = int(list(te.sort_values("MAE").modelo).index(hw)) + 1
    hw_better_lin = thw.MAE < tlin.MAE
    hw_better_pers = thw.MAE < pers.MAE
    fxhw, fxlin = _row(fx, hw), _row(fx, lin)
    txt = []
    txt.append(f"En validación, el candidato Holt-Winters con menor MAE fue **{hw}** (MAE {f(float(_row(o['validation'], hw).MAE))}), "
               f"y la regresión lineal seleccionada fue **{lin}** (MAE {f(float(_row(o['validation'], lin).MAE))}). La selección usó solo validación.")
    txt.append(f"En prueba (pronóstico de 1 hora con parámetros congelados) **{hw}** obtuvo MAE {f(thw.MAE)}, RMSE {f(thw.RMSE)}, "
               f"MAPE {f(thw.MAPE_pct)} % y R² {f(thw.R2, 4)}; **{lin}** obtuvo MAE {f(tlin.MAE)} y R² {f(tlin.R2, 4)}. "
               f"Holt-Winters {'mejora' if hw_better_lin else 'no mejora'} a la regresión lineal en MAE y "
               f"{'supera' if hw_better_pers else 'no supera'} a la persistencia de 1 h (MAE {f(pers.MAE)}); "
               f"el baseline estacional de 24 h tuvo MAE {f(s24.MAE)}.")
    txt.append(f"El mejor modelo por MAE en prueba, solo a modo descriptivo, fue **{best.modelo}** (MAE {f(best.MAE)}); Holt-Winters quedó en la posición {rank_hw} de {len(te)}.")
    txt.append(f"Con origen fijo (144 h hacia adelante sin observar datos nuevos) el MAE de Holt-Winters fue {f(fxhw.MAE)} y el de la regresión lineal recursiva {f(fxlin.MAE)}. "
               "Es una tarea más difícil que la de una hora; sus métricas no deben mezclarse.")
    if thw.R2 < 0:
        txt.append("Un R² negativo indica un error cuadrático mayor que el de predecir con la media real del bloque de prueba; esa media no se conoce al pronosticar, "
                   "así que el valor debe leerse como señal de dificultad (cambio de nivel, muchos conteos pequeños), no como un fallo del cálculo.")
    alertas = []
    val = o["validation"]; rank_val = int(list(val.modelo).index(hw)) + 1
    if val.iloc[0].modelo != hw:
        alertas.append(f"**Selección en validación:** el mejor modelo global fue **{val.iloc[0].modelo}** (MAE {f(float(val.iloc[0].MAE))}); "
                       f"{hw} fue el mejor de su familia pero ocupó la posición {rank_val} de {len(val)} (MAE {f(float(_row(val, hw).MAE))}). "
                       "Su resultado en prueba no se anticipaba en validación, así que no debe leerse como superioridad demostrada.")
    pt = o["predictions"]; yt = pt.loc[pt.bloque == "test", "real"]
    mae0 = float(yt.abs().mean())
    if mae0 < float(te.MAE.min()):
        alertas.append(f"**Referencia trivial:** pronosticar siempre 0 daría MAE {f(mae0)} en prueba (la media del bloque; la mediana es {f(float(yt.median()))}), "
                       f"menor que el del mejor modelo ({best.modelo}, {f(best.MAE)}). Ningún modelo supera esa referencia en MAE: con conteos concentrados cerca de cero "
                       "el MAE premia a los pronósticos bajos. Por eso se reportan también R², el sesgo y la correlación con lo real.")
    cvg = o["cv"].groupby("modelo").MAE.agg(["mean", "max"])
    if hw in cvg.index and lin in cvg.index and cvg.loc[hw, "mean"] > 1.5 * cvg.loc[lin, "mean"]:
        alertas.append(f"**Inestabilidad en validación cruzada:** {hw} tuvo MAE medio {f(cvg.loc[hw, 'mean'], 0)} (peor bloque {f(cvg.loc[hw, 'max'], 0)}) "
                       f"frente a {f(cvg.loc[lin, 'mean'], 0)} de {lin}. "
                       + ("Las variantes sobre log(1+y) se retrotransforman con expm1, que amplifica de forma exponencial los errores tras un cambio de régimen; "
                          "las variantes sin transformación (p. ej. multiplicativa) no mostraron ese problema. " if "log1p" in hw else "")
                       + "La ventaja en un único bloque de prueba no es evidencia suficiente para uso operativo.")
    return {"texto": txt, "alertas": alertas, "hw": hw, "lin": lin, "thw": thw, "tlin": tlin, "pers": pers, "best": best}


def hypotheses_text(o: dict) -> list[str]:
    h = o["hypotheses"]
    out = []
    for _, r in h.iterrows():
        dec = "se rechaza H0" if r.rechaza_H0_alpha_005 else "no se rechaza H0"
        out.append(f"**{r.hipotesis}**: estadístico {f(r.estadistico)} (gl={int(r.gl)}), p ajustado por Holm {p_fmt(r.p_Holm)} → {dec} con α = 0,05.")
    lb = o["ljungbox_hw"].iloc[0]
    out.append(f"Diagnóstico de residuos de {lb.modelo} (Ljung-Box, 24 rezagos, entrenamiento): Q = {f(lb.Q_24)}, p {p_fmt(lb.p)}. "
               + ("Persiste autocorrelación en los residuos: el modelo no captura toda la dependencia." if lb.p < .05 else "No hay evidencia de autocorrelación residual remanente."))
    return out
