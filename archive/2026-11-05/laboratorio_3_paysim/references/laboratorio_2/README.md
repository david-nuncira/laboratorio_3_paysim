# Laboratorio 2 - PaySim adaptado

Estadística para Analítica de Datos. Profesor: Duván Cataño. UPTC, 2026-II.
Equipo de David Nuncira, Juan Valderrama y Daniel Carrillo 

El análisis usa el CSV real del Laboratorio 1: **6.362.620 filas, 7 columnas y 743 horas**. El ZIP completo contiene el dataset en `data/transacciones.csv`. Al abrir los notebooks y el dashboard se presentan resultados reales guardados, sin cargar los 385.7 MB de nuevo.

## Inicio rápido en Windows / VSCode

Extraer el ZIP, abrir la carpeta `laboratorio_2_paysim` en VSCode y abrir una terminal PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

La aplicación se abre en la dirección local que muestra Streamlit. No requiere cuenta, API, credenciales ni acceso a internet una vez instaladas las dependencias. Para notebooks, instalar las extensiones oficiales Python y Jupyter de VSCode y seleccionar el intérprete `.venv`.

## Recalcular todo

```powershell
.\.venv\Scripts\python.exe scripts/run_analysis.py --data data/transacciones.csv
.\.venv\Scripts\python.exe scripts/create_report.py
.\.venv\Scripts\python.exe scripts/build_notebooks.py
.\.venv\Scripts\python.exe scripts/verify.py
```

También se admite `--data "C:\carpeta\transacciones.csv"`. El archivo debe tener, en este orden: `id_transaccion,step,tipo_transaccion,monto,cuenta_origen,cuenta_destino,tipo_cuenta_destino`. Separador coma y punto decimal. La huella SHA-256 y las versiones exactas del análisis están en `results/analysis_summary.json`.

Python recomendado: 3.11 o superior. Verificado aquí con Python 3.14.7; las versiones instaladas se registran en `requirements-lock.txt`. Para replicar ese entorno exacto, usar Python 3.14 e instalar ese archivo. El cálculo completo puede tardar varios minutos y consumir varios GB de RAM; se recomienda equipo de 8 GB como mínimo, preferiblemente 16 GB. Los resultados y el dashboard precalculado requieren mucho menos.

## Estructura

- `notebooks/00_Contexto_y_reproduccion.ipynb`: población, unidad, esquema, calidad y recálculo opcional.
- `notebooks/01_Descriptivas_Forma_IQR.ipynb`: centro, dispersión, posición, forma, IQR y gráficos.
- `notebooks/02_Asociaciones_e_Hipotesis.ipynb`: correlaciones, contingencias, MI, normalidad, Levene, ANOVA, Welch y Kruskal.
- `notebooks/03_Sintesis_y_Estacionalidad.ipynb`: objetivo horario, ACF, ADF y conexión SARIMA.
- `src/engine.py`, `src/plots.py`, `src/narrative.py`: cálculos, figuras e interpretaciones compartidos.
- `app.py`: navegación por los siete apartados, tablas, gráficos interactivos y descargas.
- `reports/`: informe PDF/Markdown, matriz de cumplimiento e identificación del grupo.
- `results/`: tablas CSV, JSON de precisión completa, 15 figuras y registro de verificación.
- `references/`: los dos PDF y cuatro notebooks docentes originales.
- `tests/`: 8 pruebas de cálculo y prueba de las 9 pantallas del dashboard e interacciones.
- `.vscode/`: configuración sugerida de intérprete y extensiones.

## Criterios estadísticos

Las descriptivas, contingencias y pruebas mixtas usan el archivo completo. Solo la correlación transaccional usa una muestra aleatoria sin reemplazo de hasta 200000 filas (semilla 42), y Shapiro hasta 5000 por grupo. Las figuras transaccionales usan hasta 100000 para legibilidad; las horarias usan todas las horas. El JSON conserva precisión interna; las tablas e informe muestran cuatro decimales. Un p menor de 0.0001 se presenta con esa cota.

Se analizan cuatro cuantitativas: step y monto (transacciones), conteo y monto promedio (horas). Como monto es la única continua original, el apartado 4 añade dos pares de continuas agregadas por hora y explica la adaptación. Para categorías se usa tipo de transacción vs tipo de destino y vs franja derivada de step % 24. No se someten cuentas individuales de alta cardinalidad a un chi-cuadrado con esperadas insuficientes.

La fase temporal no es un reloj civil confirmado. El CV de step depende del origen y no sirve para comparar variabilidad sustantiva. Kruskal compara rangos/distribuciones y solo bajo formas similares admite lectura de diferencias de medianas. Welch complementa la comparación de medias con varianzas desiguales. Los p son exploratorios y nominales por la dependencia temporal y cuentas repetidas.

El entregable resuelve EDA y asociación; la conexión con SARIMA plantea el modelado del siguiente laboratorio. No se atribuye un rendimiento de pronóstico sin haberlo medido. Las variables de monto de la misma hora son contemporáneas y no deben usarse para predecir esa hora como si se conocieran de antemano.

## Verificaciones

`results/verification.json` registra los comandos ejecutados y sus resultados. Los notebooks incluyen verificaciones independientes de covarianza, Pearson, frecuencias esperadas, chi-cuadrado, continuidad horaria y ACF. Los datos sintéticos de prueba se crean temporalmente y no forman parte de los resultados del laboratorio.

La copia original se localizó en `Downloads/transacciones.csv`; no se modificó. Las rutas `/mnt/data/` mencionadas en el chat anterior no corresponden a este equipo Windows. Los seis adjuntos se recuperaron de ese chat y se conservaron en `references/`. No se recuperó un informe adicional del Laboratorio 1, por lo que no se afirma haberlo leído.
