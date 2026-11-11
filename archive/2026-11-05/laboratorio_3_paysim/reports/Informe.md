# Laboratorio 3 · PaySim adaptado

David Nuncira · Juan Valderrama · Daniel Carrillo

## 1. Introducción y objetivo

Este Laboratorio 3 continúa el PaySim adaptado de los Laboratorios 1 y 2. Equipo: David Nuncira, Juan Valderrama y Daniel Carrillo, nombres respaldados por el README de GitHub. Asignatura: Estadística para Analítica de Datos; profesor Duván Cataño, UPTC, 2026-II.

Objetivo: pronosticar el conteo de transacciones de la próxima hora simulada usando regresión lineal supervisada. La unidad cambia de transacción (6.362.620 filas, 7 columnas) a hora (743 observaciones). El objetivo es cuantitativo discreto, por lo que se elige la rama de regresión de la guía, no regresión logística. Se comparan modelos lineales con y sin variables temporales y cinco referencias simples. No se atribuyen etiquetas de fraude al dataset adaptado.

La guía adjunta exige nueve apartados en notebook y una aplicación Streamlit con descripción, visualizaciones, predicciones e inputs. Todos están implementados. SARIMA se deja como trabajo posterior opcional: el requisito de regresión lineal queda cubierto sin sustituirlo por SARIMA.

## 2. Descripción de variables

Esquema original: id_transaccion (identificador nominal), step (hora simulada discreta), tipo_transaccion (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER), monto (continua en unidades del simulador), cuenta_origen y cuenta_destino (identificadores nominales), tipo_cuenta_destino (CLIENTE/COMERCIO). Se conserva el diccionario del Laboratorio 2 como fuente.

Variable objetivo y_t = transacciones_hora, conteo por step. Predictores: lag_1=y_(t-1), lag_24=y_(t-24). Variante temporal: step (tendencia conocida), sin(2πt/24), cos(2πt/24), sin(2πt/12), cos(2πt/12). La fase es step % 24; no se asigna día de semana, festivos ni una fecha civil sin evidencia. El periodo 168 se usa como referencia de repetición cada siete ciclos diarios, sin afirmar calendario semanal real.

Disponibilidad: todos los predictores existen antes de observar y_t. Se excluyen monto_total, monto_promedio y monto_mediano de la misma hora, porque se conocen al finalizarla y producirían fuga. Identificadores no aportan una señal temporal defendible.

## 3. Limpieza y preprocesamiento

Fuente efectiva de cálculo: results/hourly_series.csv recuperado mediante GitHub conectado. Se validan tipos numéricos, finitud, enteros no negativos, unicidad y continuidad de step. Se ordena cronológicamente y se construyen rezagos con shift positivo. Se pierden las primeras 24 horas por falta de historia y quedan 719 filas supervisadas. No se sustituyen horas ausentes por cero ni se eliminan picos válidos por IQR.

Partición fijada antes de comparar modelos: entrenamiento steps 25-455 (431 filas), validación 456-599 (144), prueba 600-743 (144). No hay barajado. StandardScaler se ajusta solo al bloque de entrenamiento dentro del pipeline en cada ajuste.

La suma de los conteos coincide con 6.362.620 y hay 743 horas, 1-743. No se recuperó el CSV original de 385.653.971 bytes. La ausencia de nulos y duplicados y los 16 montos cero transaccionales son hallazgos guardados del Laboratorio 2, no una nueva auditoría de millones de filas. La huella original registrada es 262ae36b53af5b4a7798a1429d389710ff6926f2e81f2d6842e1b3626765f684. El ZIP anterior inspeccionado no contiene transacciones.csv.

## 4. Análisis exploratorio de datos (EDA)

La serie completa muestra asimetría, muchos conteos pequeños, picos y una fuerte caída de actividad en la segunda mitad. Se conservan tablas globales y por bloques de 168 horas para describir ese cambio. Los perfiles y distribuciones empleados para discutir predictores se calculan sobre entrenamiento; la serie completa se presenta para documentar el alcance y el desplazamiento observado, sin ajustar un punto de corte al test.

El Laboratorio 2 registró media horaria 8.563,42, mediana 529, mínimo 2, máximo 51.352, ACF(1)=0,9235 y ACF(24)=0,7546. Estas ACF son descriptivas de toda la serie y no garantizan generalización. El nuevo código recalcula descriptivas y genera gráficos de volumen, perfil de fase, distribución, predicciones y errores.

La caída de nivel hace que un ajuste global extrapole mal sobre las horas de baja actividad. Escalar no elimina cambios de régimen, ni transforma los conteos en normales. Las gráficas no justifican causalidad.

## 5. Pruebas de hipótesis e interpretación

Se plantean dos contrastes exploratorios, con α=0,05, usando únicamente entrenamiento y el modelo lineal temporal sin recorte. H0_1: los coeficientes de lag_1 y lag_24 son simultáneamente cero, condicionados por variables temporales. H0_2: los cinco coeficientes temporales son simultáneamente cero, condicionados por los rezagos.

Se estima OLS con covarianza HAC/Newey-West, maxlags=24 y corrección de muestra; luego se aplican contrastes Wald y ajuste Holm para dos hipótesis. Se rechazan ambas H0; para rezagos el p numérico se redondea a cero por subdesbordamiento (no probabilidad exactamente nula); para temporales p ajustado≈8,92×10^-15. Los coeficientes y sus IC95% HAC se exportan. La significación temporal dentro de train no garantiza mejora futura: la variante temporal empeoró el MAE de validación.

HAC contempla heterocedasticidad y dependencia de corto plazo bajo supuestos asintóticos; no resuelve cambios estructurales, endogeneidad ni estabilidad. Ljung-Box de residuos a 24 horas es diagnóstico adicional, con p nominal y sin usarlo como tercer contraste confirmatorio. Los hallazgos transaccionales del Laboratorio 2 se conservan en references, separados de estas pruebas nuevas.

## 6. Construcción de pipelines

Para cada candidato: selección explícita de columnas -> StandardScaler -> LinearRegression con intercepto. Las filas con rezagos iniciales faltantes se excluyen antes de la partición; no hay imputación con información futura. El escalamiento se aprende dentro del pipeline para evitar fuga. En OLS sin regularización escalar no debe mejorar por sí solo el rendimiento: aporta coherencia al proceso y estabilidad numérica.

Modelo A: β0 + β1 lag_1 + β24 lag_24. Modelo B: añade step y armónicos de 24 y 12 horas. La regla de salida max(0, y_estimado) se define antes de evaluar, porque el objetivo es un conteo no negativo; el modelo ajustado sigue siendo regresión lineal, con posprocesamiento explícito. Se reportan también métricas sin recorte para auditar esta decisión. No se redondean predicciones al calcular errores.

Las transformaciones temporales son deterministas, conocidas anticipadamente. Los datos reales observados de horas previas se permiten en evaluación de una hora. Un horizonte fijo utiliza pronósticos recursivos y jamás inserta los conteos reales futuros.

## 7. Entrenamiento de modelos

Primero se realiza validación temporal expansiva dentro de entrenamiento: tres bloques de 72 horas con train creciente. Los resultados quedan en cv_metrics.csv; sirven para describir estabilidad interna. Los dos candidatos y la regla de recorte son fijos, sin búsqueda de hiperparámetros. Se elige el candidato lineal por MAE en validación: gana lineal_lags (1.938,37 frente a 2.304,44 del temporal). La variante temporal se conserva como contraste requerido y su deterioro se informa explícitamente.

Baselines: media del entrenamiento; persistencia de la hora anterior; repetición de hace 24 horas; repetición de hace 168 horas; promedio de hasta siete observaciones previas de la misma fase diaria. En validación y prueba hay historia suficiente para los siete ciclos. El baseline elegido por MAE de validación es estacional_24 (1.780,71): ya supera al lineal en ese bloque.

Después se reajustan candidatos con train+validación (steps 25-599), se congelan parámetros y se evalúa test una sola vez como protocolo final, sin cambiar selección por lo observado allí. La aplicación reajusta el candidato lineal elegido con las 719 filas, separado del ajuste utilizado para medir test. Se muestran otros candidatos en test como comparación descriptiva, sin reselección.

## 8. Evaluación del modelo

Protocolo principal: evaluación secuencial de una hora, 144 predicciones en steps 600-743. Se conocen los conteos anteriores hasta t-1; los parámetros no se actualizan dentro del bloque. Para lineal_lags: MAE=1,040.06, RMSE=1,702.59, MAPE=7,379.64% y R²=-0.2222. WAPE=144.58%. Las tablas conservan precisión completa.

Frente al baseline estacional_24 seleccionado en validación, el MAE de prueba baja 26.83% (1.421,38 a 1.040,06). Sin embargo, persistencia_1 obtiene MAE 1.003,33, menor que el lineal, aunque su RMSE es mayor (1.800,72). No puede afirmarse que la regresión gane a todos los baselines. R² negativo significa SSE mayor que la referencia retrospectiva de la media real del bloque; esa media test no sería conocida para pronosticar.

MAPE=100·media(|y-ŷ|/|y|) solo para y distinto de cero; MAPE_n informa la cantidad válida (144 en test). Aquí no hay ceros, pero hay muchos conteos mínimos, que inflan el porcentaje. MAE mide error absoluto típico en transacciones; RMSE penaliza más los errores grandes. WAPE=100·Σ|error|/Σ|y| complementa MAPE, sin ocultar errores grandes.

Contraste a origen fijo: se predicen todas las 144 horas desde step 599 con rezagos recursivos. El MAE lineal sube a 1.963,35, RMSE a 2.242,74 y R² baja a -1,1207. La repetición recursiva de las últimas 24 horas obtiene MAE 2.047,74. Es una tarea diferente y más difícil; las métricas de una hora no deben publicitarse como rendimiento de 144 horas. No se ofrecen intervalos de incertidumbre sin validación.

## 9. Conclusiones

Se construyó un proceso reproducible de regresión lineal supervisada sobre el mismo objetivo horario de los laboratorios anteriores. Los rezagos de 1 y 24 horas son conocidos a tiempo de predicción y están asociados al objetivo en entrenamiento. Añadir variables temporales conocidas no mejoró el MAE de validación, pese a su significación en el contraste exploratorio; se conserva la variante simple para predicción.

El rendimiento absoluto es insuficiente para uso operativo: R² negativo, MAPE y WAPE muy altos y dependencia del régimen de actividad. La mejora frente al baseline estacional elegido en validación no equivale a superioridad universal. La persistencia de una hora debe mantenerse como referencia fuerte. La selección lineal responde al objetivo académico, no a una afirmación de ser el mejor modelo de producción.

El horizonte recursivo acumula errores y no reemplaza una evaluación específica por horizonte. Próximos pasos: más datos que cubran distintos regímenes, ventanas móviles y detección de cambios, regresión de conteos o transformación del objetivo con corrección adecuada y, como contraste opcional, SARIMA. Estas alternativas requerirían nueva validación temporal antes de cualquier afirmación de mejora.

La app permite filtrar horas, comparar modelos, introducir rezagos manuales y pronosticar de 1 a 72 horas. No se modificó el repositorio remoto. Los resultados anteriores se preservan en references/laboratorio_2; el CSV original no se distribuye porque no está disponible. El proyecto funciona sin él usando la serie agregada de 743 filas.