# Laboratorio 3 · PaySim adaptado — Regresión lineal + Holt-Winters + dashboard Streamlit

Estadística para Analítica de Datos · UPTC · 2026-II · Prof. Duván Cataño  
David Nuncira, Juan Valderrama y Daniel Carrillo.

## Datos (igual que en el Laboratorio 2)
`results/hourly_series.csv` (la serie del Laboratorio 2) ya viene incluida; resultados, informe y notebook ya están calculados con ella.
Columnas: `step, transacciones_hora, monto_total, monto_promedio, monto_mediano, hora_dia` (solo `step` y `transacciones_hora` son indispensables).
Alternativas: `data/hourly_series.csv`, la variable de entorno `LAB_SERIES`, o `data/transacciones.csv` (se agrega por `step`).

## Inicio rápido (Windows)
1. `Ejecutar_Todo.cmd` → instala, verifica, calcula resultados/figuras, genera el informe y el notebook.
2. `Iniciar_Streamlit.cmd` → abre el dashboard.

Manual:
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/verify.py
.venv\Scripts\python.exe scripts/run_analysis.py
.venv\Scripts\python.exe scripts/build_report.py
.venv\Scripts\python.exe scripts/build_notebook.py
.venv\Scripts\python.exe scripts/execute_notebook.py   # ejecuta el notebook y guarda sus salidas (sin Jupyter)
.venv\Scripts\python.exe -m streamlit run app.py
```
Notebook: abrir `notebooks/Laboratorio_3.ipynb`, elegir el entorno `.venv` y *Run All* (el notebook recalcula todo; para guardar salidas, ejecútalo y guarda).

## Estructura
- `app.py`: dashboard (filtros laterales, KPIs, Plotly, Holt-Winters configurable, comparación, pronóstico 1-72 h, descargas).
- `src/holtwinters.py`: Holt-Winters desde cero (aditivo/multiplicativo, tendencia amortiguada, log1p).
- `src/modeling.py`: pipeline lineal, baselines, métricas, HAC/Wald/Holm, ACF, Ljung-Box, descomposición.
- `src/pipeline.py`: partición, selección en validación, prueba final, CV expansiva, hipótesis, pronóstico.
- `src/narrative.py`: texto interpretativo generado con los resultados.
- `scripts/`: `run_analysis.py`, `verify.py`, `build_report.py`, `build_notebook.py`, `execute_notebook.py`.
- `reports/Informe_Laboratorio_3.md`: se genera con `build_report.py` (cifras reales de tu serie).

## Diseño metodológico
Partición cronológica 60/20/20 (train 25-455, validación 456-599, test 600-743 para 743 horas; Holt-Winters también usa las horas 1-24).
Selección solo por MAE de validación; modelos reajustados con train+validación para la prueba única. Protocolo principal: pronóstico de 1 h con
parámetros congelados (estados/rezagos actualizados con lo observado) y, aparte, origen fijo a horizonte completo. Sin intervalos de predicción (no validados).
