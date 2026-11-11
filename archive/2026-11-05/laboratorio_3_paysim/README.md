# Laboratorio 3 - PaySim adaptado

Estadística para Analítica de Datos · UPTC · 2026-II · Profesor Duván Cataño.

Equipo de tres estudiantes: **David Nuncira, Juan Valderrama y Daniel Carrillo**. Identificación respaldada por el README del [repositorio anterior](https://github.com/david-nuncira/laboratorio_2_paysim).

Proyecto completo de regresión lineal supervisada para pronosticar el conteo horario. Continúa los Laboratorios 1 y 2: 6.362.620 transacciones agregadas en 743 horas; esquema original de siete columnas. El CSV transaccional no se recuperó ni se requiere para ejecutar esta entrega. No se modificó GitHub.

## Iniciar en Windows / VSCode

Extraer el ZIP completo y abrir `Iniciar_Streamlit.cmd` con doble clic. Este archivo crea el entorno, instala las bibliotecas si faltan y abre Streamlit. También puedes abrir la carpeta `laboratorio_3_paysim` en VSCode y ejecutar:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Python 3.12 recomendado (ejecución verificada con 3.12.14). En otros sistemas: `python -m venv .venv`, activar el entorno correspondiente y usar `python -m streamlit run app.py`. Para reproducir las versiones verificadas, instalar `requirements-lock.txt` en vez del archivo general.

En VSCode, instalar las extensiones Python y Jupyter. Si ya estaba seleccionado otro Python, presionar Ctrl+Shift+P, elegir `Python: Select Interpreter` y seleccionar `.venv/Scripts/python.exe`. Para abrir la app usar el comando Streamlit anterior o F5 con la configuración `Abrir PaySim con Streamlit`; el botón de ejecutar un archivo Python directamente no inicia el servidor Streamlit. Para el notebook abrir `notebooks/Laboratorio_3.ipynb`, seleccionar el mismo entorno y ejecutar todas las celdas. El notebook incluye resultados guardados y también recalcula las tablas y figuras.

En Colab subir el ZIP del proyecto, extraerlo en `/content`, abrir el notebook y ejecutar una celda previa:

```python
%pip install -r /content/laboratorio_3_paysim/requirements.txt
```

El notebook busca `/content/laboratorio_3_paysim` automáticamente. La ejecución se verificó en Windows; Colab se documenta como ruta razonable, sin afirmar una prueba allí. No requiere descargar millones de transacciones, credenciales ni conexión una vez instaladas las bibliotecas.

## Recalcular y comprobar

```powershell
.\.venv\Scripts\python.exe scripts/run_analysis.py
.\.venv\Scripts\python.exe scripts/build_deliverables.py
.\.venv\Scripts\python.exe scripts/verify.py
```

`run_analysis.py` calcula métricas, hipótesis, coeficientes HAC, CV, predicciones y figuras. `build_deliverables.py` regenera el notebook sin salidas, el texto y el PDF. Después ejecutar el notebook para guardar sus salidas. La verificación prueba datos, ausencia de fuga, fórmulas de métricas y pantallas e interacciones de Streamlit. El resumen final de ejecución se guarda en `results/execution_record.json`.

## Método y resultados reales

- Objetivo: `transacciones_hora`; predictores principales `lag_1` y `lag_24`. Variante adicional con tendencia y armónicos de 24 y 12 horas conocidos anticipadamente.
- Pipeline: `StandardScaler -> LinearRegression`, ajustado solo con el entrenamiento de cada bloque. Posprocesamiento predefinido: recorte de predicciones negativas a cero; también se publican resultados sin recorte.
- Partición cronológica: train 25-455, validación 456-599, test 600-743. Se excluyen las primeras 24 horas por falta de rezagos. CV expansiva de tres bloques dentro de train. Selección por MAE de validación, nunca por test.
- La variante temporal obtuvo peor MAE de validación; se seleccionó `lineal_lags`. Las hipótesis con covarianza HAC y corrección Holm solo utilizan train.
- Baselines: media de train, persistencia 1 h, estacional 24 h, estacional 168 h, promedio de siete ciclos de 24 h. El periodo 168 no implica un calendario civil confirmado.
- Test secuencial de una hora del lineal: **MAE 1.040,06; RMSE 1.702,59; MAPE 7.379,64%; R² -0,2222**. El baseline 24 h seleccionado en validación obtuvo MAE 1.421,38. Persistencia 1 h logró MAE 1.003,33: la regresión no gana a todos los baselines.
- Origen fijo de 144 h: MAE lineal 1.963,35. No confundir este escenario con la evaluación secuencial que incorpora observaciones previas ya disponibles.

R² negativo y errores porcentuales muy grandes muestran límites importantes. Hay muchos conteos pequeños y un cambio de nivel entre bloques. La entrega satisface el objetivo académico; no valida un uso operativo preciso. No se inventan intervalos ni se asume normalidad del conteo. SARIMA queda como contraste opcional futuro, no como reemplazo del modelo exigido.

## Archivos

- `notebooks/Laboratorio_3.ipynb`: nueve apartados, código ejecutable y salidas verificadas.
- `app.py`: proyecto, filtros de horas, gráficos interactivos, evaluación, inputs manuales, pronóstico recursivo 1-72 h y descargas.
- `src/modeling.py`: funciones compartidas para notebook y app.
- `scripts/`: cálculo, generación de entregables y verificación.
- `reports/Informe_Laboratorio_3.pdf` y `Informe.md`: explicación, resultados, interpretación y limitaciones.
- `results/`: datos horarios, tabla supervisada, métricas de validación/test/CV, predicciones, hipótesis, coeficientes, diagnósticos y gráficos.
- `references/Laboratorio_3.pdf`: guía original recuperada del chat.
- `references/provenance.json`: rutas y huellas de blobs consultados en GitHub.
- `references/laboratorio_2/`: copia preservada de resultados, cuatro notebooks y README previos, contrastada con GitHub para los notebooks.
- `references/lab2_analysis_summary.json` y `README_GitHub.md`: documentos recuperados directamente del conector GitHub.

Los resultados nuevos se escriben en `results/`; los previos permanecen separados e intactos. No se incluye el entorno de Python en el ZIP.

## Límites de las fuentes

La ruta `/mnt/data/Laboratorio_3.pdf` era del chat anterior; en este equipo Windows se recuperó el adjunto mediante la conversación referenciada. La serie y el resumen se recuperaron de GitHub. Los notebooks locales de la entrega anterior se cotejan por SHA de blob con los cuatro consultados por el conector. El ZIP previo encontrado no contiene el CSV original; no se recalculó limpieza ni estadística transaccional. No se recuperó un informe independiente del Laboratorio 1. Las cifras de ese nivel se citan como evidencia previa del Laboratorio 2 y se preserva su huella original.

Documentación técnica: [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [covarianza robusta HAC](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html).
