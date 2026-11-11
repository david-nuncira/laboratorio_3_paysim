"""Genera reports/Informe_Laboratorio_3.md con las cifras reales de results/ (ejecutar run_analysis.py antes)."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data import load_series, TARGET
from src.pipeline import run_all
from src.narrative import f, md_table, findings, hypotheses_text, p_fmt, diag_test


def build():
    d = load_series(); o = run_all(d); F = findings(o); sp = o["splits"]
    ds = o["descriptive"].iloc[0]; ac = o["acf_resumen"]; hw = o["selected"]["holt_winters"]
    tcols = ["modelo", "familia", "MAE", "RMSE", "MAPE_pct", "WAPE_pct", "R2"]
    L = []
    A = L.append
    A("# Laboratorio 3 · Pronóstico del volumen horario de transacciones con regresión lineal y Holt-Winters\n")
    A("**Estadística para Analítica de Datos · UPTC · 2026-II · Profesor Duván Cataño**  \n**Equipo:** David Nuncira, Juan Valderrama y Daniel Carrillo\n")
    A("> Resumen. Se continúa el PaySim adaptado de los Laboratorios 1 y 2. La unidad de análisis pasa de la transacción a la hora simulada y "
      "la variable objetivo es `transacciones_hora`. Se construyen pipelines de regresión lineal (rezagos y variables temporales), cinco baselines y "
      "el modelo **Holt-Winters** (suavizamiento exponencial triple, periodo 24 h) y se evalúan con partición cronológica y métricas MAE, RMSE, MAPE y R². "
      "Se entrega además un dashboard interactivo en Streamlit.\n")
    A("## 1. Introducción y objetivo\n")
    A(f"El Laboratorio 2 mostró una serie horaria de {len(d)} observaciones ({f(ds.suma, 0)} transacciones) con autocorrelación alta (ACF(1) = {f(ac['acf_1'], 4)}; "
      f"ACF(24) = {f(ac['acf_24'], 4)}) y un perfil intradiario marcado. **Objetivo:** pronosticar el conteo de transacciones de las próximas horas. "
      "Como la variable es un conteo (cuantitativa discreta) se usa la rama de **regresión** de la guía; la regresión logística no aplica porque el dataset adaptado no incluye una etiqueta de clase. "
      "Por la fuerte estacionalidad se incorpora Holt-Winters, exigido para la sección de pronósticos.\n")
    A("## 2. Descripción de variables\n")
    A("| Variable | Tipo | Descripción |\n|---|---|---|\n| `step` | discreta | hora simulada (1 a N) |\n| `transacciones_hora` | conteo (objetivo) | número de transacciones en la hora |\n"
      "| `monto_total`, `monto_promedio`, `monto_mediano` | continuas | agregados de `monto` de la misma hora (no se usan como predictores: se conocen al cierre de la hora → fuga) |\n"
      "| `fase_24` | discreta | `step % 24`, posición dentro del ciclo diario simulado |\n| `lag_1`, `lag_24` | conteo | objetivo 1 y 24 horas antes (disponibles al pronosticar) |\n"
      "| `step`, `sin/cos(2πt/24)`, `sin/cos(2πt/12)` | deterministas | tendencia y armónicos conocidos de antemano |\n")
    A("No se asigna día de semana ni fecha civil: el dataset es simulado y no hay evidencia de calendario real.\n")
    A("## 3. Limpieza y preprocesamiento\n")
    A(f"La serie se lee igual que en el Laboratorio 2 (`results/hourly_series.csv`). Se validan finitud, valores no negativos, unicidad y continuidad de `step`. "
      f"No se imputan horas ni se eliminan picos por IQR (son actividad válida). Para la regresión se pierden las primeras 24 horas por falta de `lag_24`. "
      "Partición cronológica 60/20/20 sobre las filas supervisadas, fijada antes de comparar modelos; Holt-Winters usa además las horas 1–24 dentro de entrenamiento.\n")
    A(md_table(o["splits_table"], 0) + "\n")
    A("## 4. Análisis exploratorio (EDA)\n")
    A(md_table(o["descriptive"].rename(columns={"suma": "total"}), 2) + "\n")
    A(f"La distribución es muy asimétrica (asimetría {f(ds.asimetria)}; CV {f(ds.cv_pct)} %) y la serie cambia de régimen entre bloques de 168 horas:\n")
    A(md_table(o["regimes"], 1) + "\n")
    A("![Serie y partición](../results/figures/serie_particion.png)\n\n![EDA](../results/figures/eda.png)\n")
    A("El perfil por fase muestra el ciclo diario que justifica un periodo estacional de 24 y la ACF decae lentamente. El cambio de nivel entre bloques implica que un ajuste global puede extrapolar mal; las gráficas describen asociación, no causalidad.\n")
    A("## 5. Pruebas de hipótesis e interpretación\n")
    A("Con α = 0,05 y solo datos de entrenamiento. H1 y H2 usan OLS con covarianza HAC/Newey-West (24 rezagos) y contraste de Wald; H3 es Kruskal-Wallis entre las 24 fases del ciclo (sin supuesto de normalidad). Se aplica corrección de Holm a las tres hipótesis.\n")
    A(md_table(o["hypotheses"], 4, pcols=("p", "p_Holm")) + "\n")
    for t in hypotheses_text(o): A("- " + t)
    A("\nLa significación dentro de entrenamiento no garantiza mejor pronóstico futuro; los p-valores son nominales por la dependencia temporal. Coeficientes con IC95 % HAC:\n")
    A(md_table(o["coef_hac"], 4, pcols=("p_normal",)) + "\n")
    A("## 6. Construcción de pipelines\n")
    A("**Regresión lineal:** selección explícita de columnas → `StandardScaler` (ajustado solo con entrenamiento) → `LinearRegression`. "
      "Modelo A (`lineal_lags`): `lag_1`, `lag_24`. Modelo B (`lineal_temporal`): A + tendencia y armónicos. Regla fija de salida: `max(0, ŷ)`.\n")
    A("**Holt-Winters** (implementación propia, `src/holtwinters.py`, verificada contra la serie sin fuga de información). Con periodo m = 24:\n")
    A("```\nŷ(t|t-1) = l(t-1) + φ·b(t-1) + s(t-m)                         (estacionalidad aditiva)\nl(t) = α·(y(t) - s(t-m)) + (1-α)·(l(t-1) + φ·b(t-1))\nb(t) = β·(l(t) - l(t-1)) + (1-β)·φ·b(t-1)\ns(t) = γ·(y(t) - l(t-1) - φ·b(t-1)) + (1-γ)·s(t-m)\n```\n")
    A("En la variante multiplicativa, `ŷ = (l + φb)·s` y las actualizaciones usan cocientes. Los parámetros α, β, γ (y φ si la tendencia es amortiguada) se estiman minimizando la suma de cuadrados de los errores de un paso en entrenamiento; "
      "los estados iniciales salen de los primeros ciclos de entrenamiento. Se compara una rejilla fija de candidatos: sin tendencia / tendencia amortiguada × estacionalidad aditiva / multiplicativa, y las dos formas aditivas sobre `log(1+y)` (con retrotransformación y recorte a cero). "
      "α cercano a 1 haría que el nivel siga casi solo la última observación; γ controla la rapidez con que se actualiza el perfil de 24 h.\n")
    A("## 7. Entrenamiento de modelos\n")
    A("Se usa validación cruzada temporal expansiva (3 bloques de 72 h dentro de entrenamiento) solo para describir estabilidad. Después se compara en validación (candidatos entrenados únicamente con *train*), "
      "se elige por MAE sin mirar *test* y, antes de la prueba final, se reajustan con *train+validación*.\n")
    A("**Validación (1 hora):**\n"); A(md_table(o["validation"][tcols], 2) + "\n")
    A("**Rejilla Holt-Winters en validación:**\n")
    gcols = [c for c in o["hw_grid_validation"].columns if c in ("modelo", "MAE", "RMSE", "R2", "origen_fijo_MAE", "param_alpha", "param_gamma", "param_beta", "param_phi")]
    A(md_table(o["hw_grid_validation"][gcols], 4) + "\n")
    cvm = o["cv"].groupby("modelo")[["MAE", "RMSE", "R2"]].mean().reset_index()
    A("**Validación cruzada expansiva (promedio de 3 bloques):**\n"); A(md_table(cvm, 2) + "\n")
    A(f"Selección: regresión lineal → `{o['selected']['lineal']}`; Holt-Winters → `{hw}`; baseline → `{o['selected']['baseline']}`.\n")
    A("## 8. Evaluación del modelo\n")
    A("Protocolo principal: pronóstico de **una hora** sobre el bloque de prueba; el estado de Holt-Winters y los rezagos incorporan lo observado hasta *t−1*, pero los parámetros no se reestiman. MAPE se calcula solo con y ≠ 0; WAPE complementa a MAPE cuando hay conteos pequeños.\n")
    A(md_table(o["test"][tcols], 2) + "\n")
    A("**Origen fijo (horizonte completo del bloque de prueba, sin observar datos nuevos):**\n"); A(md_table(o["test_origen_fijo"][tcols], 2) + "\n")
    A("**Diagnóstico complementario en prueba (1 h): sesgo medio = real − pronóstico y correlación con lo real:**\n"); A(md_table(diag_test(o), 3) + "\n")
    A("![Prueba](../results/figures/test_pronosticos.png)\n\n![Comparación MAE](../results/figures/comparacion_mae.png)\n")
    for t in F["texto"] + F["alertas"]: A("- " + t)
    A("\n![Componentes Holt-Winters](../results/figures/hw_componentes.png)\n")
    A("**Pronóstico a 72 horas** con el modelo Holt-Winters seleccionado, ajustado con toda la serie:\n")
    A("![Pronóstico 72 h](../results/figures/hw_pronostico_72h.png)\n")
    A("## 9. Conclusiones\n")
    for t in F["texto"][:3] + F["alertas"]: A("- " + t)
    A("- Los resultados son pronósticos puntuales; no se publican intervalos porque no se validó su cobertura.")
    A("- Limitaciones: serie simulada y corta (pocos ciclos de 24 h), cambio de nivel entre bloques, muchos conteos pequeños que inflan MAPE y un único bloque de prueba (la comparación entre modelos con diferencias pequeñas no es concluyente).")
    A("- Trabajo futuro: ventanas móviles con reestimación, detección de cambios de régimen, modelos de conteo y SARIMA como contraste.\n")
    A("## Apéndice · Reproducibilidad\n")
    A("```\npython -m pip install -r requirements.txt\npython scripts/run_analysis.py\npython scripts/verify.py\npython scripts/build_report.py\npython -m streamlit run app.py\n```\n")
    out = ROOT / "reports" / "Informe_Laboratorio_3.md"
    out.parent.mkdir(exist_ok=True); out.write_text("\n".join(L), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
