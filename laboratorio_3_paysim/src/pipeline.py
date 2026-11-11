"""Orquestación completa del Laboratorio 3 (usada por notebook, scripts y Streamlit)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from .data import TARGET, M, make_splits, block_of
from .holtwinters import HoltWinters, HWConfig, hw_candidates
from .modeling import (LAGS, CALENDAR, metrics, make_features, fit_linear, predict_linear, linear_coefficients,
                       baseline_predictions, ols_hac, wald, holm, acf, ljung_box)

BASELINES = ["media_train", "persistencia_1", "estacional_24", "estacional_168", "media_estacional_7"]


def recursive_linear(model, cols, hist: np.ndarray, h: int, step0: int) -> np.ndarray:
    """Pronóstico recursivo de h horas con el lineal: cada predicción alimenta los rezagos siguientes."""
    buf = list(hist[-M:]); out = []
    for i in range(h):
        t = step0 + i
        ang = 2 * np.pi * t / 24
        row = {"lag_1": buf[-1], "lag_24": buf[-M], "step": t, "sin24": np.sin(ang), "cos24": np.cos(ang),
               "sin12": np.sin(2 * ang), "cos12": np.cos(2 * ang)}
        p = float(predict_linear(model, pd.DataFrame([row]), cols)[0])
        out.append(p); buf.append(p)
    return np.array(out)


def hw_evaluate(y, cfg: HWConfig, i_train: int, i_val: int):
    """Ajusta con y[:i_train] y evalúa en y[i_train:i_val] (un paso con parámetros fijos y origen fijo)."""
    m = HoltWinters(cfg).fit(y[:i_train])
    one = m.one_step(y)[i_train:i_val]
    multi = m.forecast(i_val - i_train, y[:i_train])
    return m, one, multi


def run_all(d: pd.DataFrame) -> dict:
    sp = make_splits(d)
    step0 = int(d.step.iloc[0]); y = d[TARGET].to_numpy(float); steps = d.step.to_numpy()
    pos = lambda s: int(s - step0 + 1)                      # nº de observaciones hasta el step s (inclusive)
    i_tr, i_va, i_te = pos(sp["train_end"]), pos(sp["val_end"]), len(y)

    x = make_features(d)
    tr, va, te = (x[x.step <= sp["train_end"]], x[(x.step > sp["train_end"]) & (x.step <= sp["val_end"])],
                  x[x.step > sp["val_end"]])
    trva = x[x.step <= sp["val_end"]]
    out = {"splits": sp, "x": x}
    out["splits_table"] = pd.DataFrame([
        {"grupo": "train (HW usa 1-%d)" % sp["train_end"], "n": len(tr), "inicio": int(tr.step.min()), "fin": int(tr.step.max())},
        {"grupo": "validacion", "n": len(va), "inicio": int(va.step.min()), "fin": int(va.step.max())},
        {"grupo": "test", "n": len(te), "inicio": int(te.step.min()), "fin": int(te.step.max())}])

    # ---------------- descriptivas y regímenes ----------------
    s = d[TARGET]
    out["descriptive"] = pd.DataFrame([{"n": len(s), "media": s.mean(), "mediana": s.median(), "std": s.std(), "min": s.min(),
                                        "max": s.max(), "asimetria": s.skew(), "curtosis_exceso": s.kurt(),
                                        "cv_pct": 100 * s.std() / s.mean(), "suma": int(s.sum())}])
    blk = (d.step - step0) // 168
    out["regimes"] = d.groupby(blk).agg(inicio=("step", "min"), fin=("step", "max"), media=(TARGET, "mean"),
                                        mediana=(TARGET, "median"), std=(TARGET, "std"), max=(TARGET, "max")).reset_index(drop=True)
    r = acf(y, 168)
    out["acf"] = pd.DataFrame({"lag": np.arange(169), "acf": r})
    out["acf_resumen"] = {"acf_1": float(r[1]), "acf_24": float(r[24]), "acf_168": float(r[168])}

    # ---------------- regresión lineal (selección por MAE de validación) ----------------
    cand_cols = {"lineal_lags": LAGS, "lineal_temporal": LAGS + CALENDAR}
    val_rows, preds = [], pd.DataFrame({"step": x.step, "real": x[TARGET]})
    lin_models = {}
    for nm, cols in cand_cols.items():
        mdl = fit_linear(tr, cols); lin_models[nm] = mdl
        val_rows.append({"modelo": nm, "familia": "regresion_lineal", **metrics(va[TARGET], predict_linear(mdl, va, cols))})
    bl_tr = baseline_predictions(x, tr)
    for nm in BASELINES:
        val_rows.append({"modelo": nm, "familia": "baseline", **metrics(va[TARGET], bl_tr.loc[va.index, nm])})

    # ---------------- Holt-Winters: rejilla en validación ----------------
    grid, hw_models = [], {}
    for cfg in hw_candidates(M, float(y.min())):
        try:
            mdl, one, multi = hw_evaluate(y, cfg, i_tr, i_va)
        except ValueError:
            continue
        hw_models[cfg.name] = (cfg, mdl)
        yv = y[i_tr:i_va]
        grid.append({"modelo": cfg.name, "familia": "holt_winters", **metrics(yv, one),
                     **{f"origen_fijo_{k}": v for k, v in metrics(yv, multi).items() if k in ("MAE", "RMSE", "R2")},
                     **{f"param_{k}": v for k, v in mdl.params_.items()}})
    hwg = pd.DataFrame(grid)
    val_rows += hwg[["modelo", "familia", "MAE", "RMSE", "MAPE_pct", "WAPE_pct", "R2", "n"]].to_dict("records")
    out["hw_grid_validation"] = hwg.sort_values("MAE").reset_index(drop=True)
    out["validation"] = pd.DataFrame(val_rows).sort_values("MAE").reset_index(drop=True)

    sel_lin = min(cand_cols, key=lambda k: float(out["validation"].set_index("modelo").loc[k, "MAE"]))
    sel_hw = str(out["hw_grid_validation"].iloc[0]["modelo"])
    sel_bl = min(BASELINES, key=lambda k: float(out["validation"].set_index("modelo").loc[k, "MAE"]))
    best_val = str(out["validation"].iloc[0]["modelo"])
    out["selected"] = {"lineal": sel_lin, "holt_winters": sel_hw, "baseline": sel_bl, "mejor_validacion": best_val}

    # ---------------- test (modelos reajustados con train+validación; una sola evaluación) ----------------
    y_te = te[TARGET].to_numpy()
    test_rows, fixed_rows = [], []
    bl_tv = baseline_predictions(x, trva)
    for nm in BASELINES:
        p = bl_tv.loc[te.index, nm].to_numpy(); preds[nm] = bl_tv[nm]
        preds.loc[x.step <= sp['val_end'], nm] = bl_tr.loc[x.step <= sp['val_end'], nm]
        test_rows.append({"modelo": nm, "familia": "baseline", **metrics(y_te, p)})
    lin_final = {}
    for nm, cols in cand_cols.items():
        mdl = fit_linear(trva, cols); lin_final[nm] = mdl
        pall = predict_linear(mdl, x, cols); preds[nm] = pall
        before_test = x.step <= sp['val_end']
        preds.loc[before_test, nm] = predict_linear(lin_models[nm], x.loc[before_test], cols)
        test_rows.append({"modelo": nm, "familia": "regresion_lineal", **metrics(y_te, predict_linear(mdl, te, cols))})
        rec = recursive_linear(mdl, cols, y[:i_va], len(te), sp["val_end"] + 1)
        fixed_rows.append({"modelo": nm, "familia": "regresion_lineal", **metrics(y_te, rec)}); preds[nm + "_origen_fijo"] = np.nan
        preds.loc[te.index, nm + "_origen_fijo"] = rec
    rep = np.tile(y[i_va - M:i_va], int(np.ceil(len(te) / M)))[:len(te)]
    fixed_rows.append({"modelo": "estacional_24", "familia": "baseline", **metrics(y_te, rep)})
    hw_final = {}
    for nm, (cfg, _) in hw_models.items():
        mdl, one, multi = hw_evaluate(y, cfg, i_va, i_te)
        hw_final[nm] = (cfg, mdl)
        full_one = mdl.one_step(y)
        preds[nm] = pd.Series(full_one, index=steps).reindex(x.step).to_numpy()
        validation_one = hw_models[nm][1].one_step(y)
        preds.loc[x.step <= sp['val_end'], nm] = pd.Series(validation_one, index=steps).reindex(x.loc[x.step <= sp['val_end'], 'step']).to_numpy()
        test_rows.append({"modelo": nm, "familia": "holt_winters", **metrics(y_te, one)})
        fixed_rows.append({"modelo": nm, "familia": "holt_winters", **metrics(y_te, multi)})
        preds[nm + "_origen_fijo"] = np.nan
        preds.loc[te.index, nm + "_origen_fijo"] = multi
    out["test"] = pd.DataFrame(test_rows).sort_values("MAE").reset_index(drop=True)
    out["test_origen_fijo"] = pd.DataFrame(fixed_rows).sort_values("MAE").reset_index(drop=True)
    preds["bloque"] = [block_of(int(s_), sp) for s_ in preds.step]
    out["predictions"] = preds

    # ---------------- validación cruzada temporal expansiva (3 bloques de 72 h dentro de train) ----------------
    cv = []
    folds = [(sp["train_end"] - 3 * 72 + 72 * k) for k in range(3)]
    cfg_sel, _ = hw_models[sel_hw]
    for k, endk in enumerate(folds, 1):
        trk, vak = x[x.step <= endk], x[(x.step > endk) & (x.step <= endk + 72)]
        for nm, cols in cand_cols.items():
            cv.append({"fold": k, "modelo": nm, **metrics(vak[TARGET], predict_linear(fit_linear(trk, cols), vak, cols))})
        _, one, _ = hw_evaluate(y, cfg_sel, pos(endk), pos(endk + 72))
        cv.append({"fold": k, "modelo": sel_hw, **metrics(y[pos(endk):pos(endk + 72)], one)})
        cv.append({"fold": k, "modelo": "persistencia_1", **metrics(vak[TARGET], vak["lag_1"])})
    out["cv"] = pd.DataFrame(cv)

    # ---------------- hipótesis (solo entrenamiento) ----------------
    cols_all = LAGS + CALENDAR
    beta, V, _ = ols_hac(tr[TARGET], tr[cols_all], 24)
    names = ["const"] + cols_all
    se = np.sqrt(np.diag(V)); z = 1.96
    out["coef_hac"] = pd.DataFrame({"variable": names, "coef": beta, "ee_HAC": se, "IC95_inf": beta - z * se,
                                    "IC95_sup": beta + z * se, "p_normal": 2 * stats.norm.sf(np.abs(beta / se))})
    w1 = wald(beta, V, [1, 2]); w2 = wald(beta, V, range(3, 8))
    groups = [tr.loc[(tr.step % 24) == f, TARGET].to_numpy() for f in range(24)]
    kw = stats.kruskal(*groups)
    ps = [w1[1], w2[1], float(kw.pvalue)]; adj = holm(ps)
    out["hypotheses"] = pd.DataFrame({
        "hipotesis": ["H1: lag_1 y lag_24 = 0 (Wald HAC)", "H2: variables temporales = 0 (Wald HAC)",
                      "H3: misma distribución en las 24 fases (Kruskal-Wallis)"],
        "estadistico": [w1[0], w2[0], float(kw.statistic)], "gl": [w1[2], w2[2], 23],
        "p": ps, "p_Holm": adj, "rechaza_H0_alpha_005": adj < 0.05})
    hw_tr = HoltWinters(cfg_sel).fit(y[:i_tr]); resid = y[:i_tr] - hw_tr.fitted_
    q, pq = ljung_box(resid[M:], 24)
    out["ljungbox_hw"] = pd.DataFrame([{"modelo": sel_hw, "Q_24": q, "p": pq, "nota": "diagnóstico, no es contraste confirmatorio"}])

    # ---------------- Holt-Winters final (toda la serie) y pronóstico ----------------
    final = HoltWinters(cfg_sel).fit(y)
    out["hw_final"] = final
    out["hw_params"] = pd.DataFrame([{"modelo": sel_hw, "m": cfg_sel.m, **final.params_}])
    out["forecast_72"] = pd.DataFrame({"step": np.arange(steps[-1] + 1, steps[-1] + 73), "pronostico_HW": final.forecast(72)})
    out["models"] = {"lin_final": lin_final, "hw_final_test": hw_final}
    return out
