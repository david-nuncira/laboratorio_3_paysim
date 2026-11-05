from pathlib import Path
import sys,json,html
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.modeling import ROOT
import pandas as pd
import nbformat as nbf
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm

def build():
    r=ROOT/'results'; summary=json.loads((r/'summary.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(r/'metrics.csv'); test=metrics[metrics.etapa=='test']
    chosen=summary['modelo_seleccionado_validacion_MAE']; baseline=summary['baseline_seleccionado_validacion_MAE']
    result=test[test.modelo==chosen].iloc[0]
    sections=[
    ('1. Introducción y objetivo', '''Este Laboratorio 3 continúa el PaySim adaptado de los Laboratorios 1 y 2. Equipo: David Nuncira, Juan Valderrama y Daniel Carrillo, nombres respaldados por el README de GitHub. Asignatura: Estadística para Analítica de Datos; profesor Duván Cataño, UPTC, 2026-II.

Objetivo: pronosticar el conteo de transacciones de la próxima hora simulada usando regresión lineal supervisada. La unidad cambia de transacción (6.362.620 filas, 7 columnas) a hora (743 observaciones). El objetivo es cuantitativo discreto, por lo que se elige la rama de regresión de la guía, no regresión logística. Se comparan modelos lineales con y sin variables temporales y cinco referencias simples. No se atribuyen etiquetas de fraude al dataset adaptado.

La guía adjunta exige nueve apartados en notebook y una aplicación Streamlit con descripción, visualizaciones, predicciones e inputs. Todos están implementados. SARIMA se deja como trabajo posterior opcional: el requisito de regresión lineal queda cubierto sin sustituirlo por SARIMA.'''),
    ('2. Descripción de variables', '''Esquema original: id_transaccion (identificador nominal), step (hora simulada discreta), tipo_transaccion (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER), monto (continua en unidades del simulador), cuenta_origen y cuenta_destino (identificadores nominales), tipo_cuenta_destino (CLIENTE/COMERCIO). Se conserva el diccionario del Laboratorio 2 como fuente.

Variable objetivo y_t = transacciones_hora, conteo por step. Predictores: lag_1=y_(t-1), lag_24=y_(t-24). Variante temporal: step (tendencia conocida), sin(2πt/24), cos(2πt/24), sin(2πt/12), cos(2πt/12). La fase es step % 24; no se asigna día de semana, festivos ni una fecha civil sin evidencia. El periodo 168 se usa como referencia de repetición cada siete ciclos diarios, sin afirmar calendario semanal real.

Disponibilidad: todos los predictores existen antes de observar y_t. Se excluyen monto_total, monto_promedio y monto_mediano de la misma hora, porque se conocen al finalizarla y producirían fuga. Identificadores no aportan una señal temporal defendible.'''),
    ('3. Limpieza y preprocesamiento', '''Fuente efectiva de cálculo: results/hourly_series.csv recuperado mediante GitHub conectado. Se validan tipos numéricos, finitud, enteros no negativos, unicidad y continuidad de step. Se ordena cronológicamente y se construyen rezagos con shift positivo. Se pierden las primeras 24 horas por falta de historia y quedan 719 filas supervisadas. No se sustituyen horas ausentes por cero ni se eliminan picos válidos por IQR.

Partición fijada antes de comparar modelos: entrenamiento steps 25-455 (431 filas), validación 456-599 (144), prueba 600-743 (144). No hay barajado. StandardScaler se ajusta solo al bloque de entrenamiento dentro del pipeline en cada ajuste.

La suma de los conteos coincide con 6.362.620 y hay 743 horas, 1-743. No se recuperó el CSV original de 385.653.971 bytes. La ausencia de nulos y duplicados y los 16 montos cero transaccionales son hallazgos guardados del Laboratorio 2, no una nueva auditoría de millones de filas. La huella original registrada es 262ae36b53af5b4a7798a1429d389710ff6926f2e81f2d6842e1b3626765f684. El ZIP anterior inspeccionado no contiene transacciones.csv.'''),
    ('4. Análisis exploratorio de datos (EDA)', '''La serie completa muestra asimetría, muchos conteos pequeños, picos y una fuerte caída de actividad en la segunda mitad. Se conservan tablas globales y por bloques de 168 horas para describir ese cambio. Los perfiles y distribuciones empleados para discutir predictores se calculan sobre entrenamiento; la serie completa se presenta para documentar el alcance y el desplazamiento observado, sin ajustar un punto de corte al test.

El Laboratorio 2 registró media horaria 8.563,42, mediana 529, mínimo 2, máximo 51.352, ACF(1)=0,9235 y ACF(24)=0,7546. Estas ACF son descriptivas de toda la serie y no garantizan generalización. El nuevo código recalcula descriptivas y genera gráficos de volumen, perfil de fase, distribución, predicciones y errores.

La caída de nivel hace que un ajuste global extrapole mal sobre las horas de baja actividad. Escalar no elimina cambios de régimen, ni transforma los conteos en normales. Las gráficas no justifican causalidad.'''),
    ('5. Pruebas de hipótesis e interpretación', '''Se plantean dos contrastes exploratorios, con α=0,05, usando únicamente entrenamiento y el modelo lineal temporal sin recorte. H0_1: los coeficientes de lag_1 y lag_24 son simultáneamente cero, condicionados por variables temporales. H0_2: los cinco coeficientes temporales son simultáneamente cero, condicionados por los rezagos.

Se estima OLS con covarianza HAC/Newey-West, maxlags=24 y corrección de muestra; luego se aplican contrastes Wald y ajuste Holm para dos hipótesis. Se rechazan ambas H0; para rezagos el p numérico se redondea a cero por subdesbordamiento (no probabilidad exactamente nula); para temporales p ajustado≈8,92×10^-15. Los coeficientes y sus IC95% HAC se exportan. La significación temporal dentro de train no garantiza mejora futura: la variante temporal empeoró el MAE de validación.

HAC contempla heterocedasticidad y dependencia de corto plazo bajo supuestos asintóticos; no resuelve cambios estructurales, endogeneidad ni estabilidad. Ljung-Box de residuos a 24 horas es diagnóstico adicional, con p nominal y sin usarlo como tercer contraste confirmatorio. Los hallazgos transaccionales del Laboratorio 2 se conservan en references, separados de estas pruebas nuevas.'''),
    ('6. Construcción de pipelines', '''Para cada candidato: selección explícita de columnas -> StandardScaler -> LinearRegression con intercepto. Las filas con rezagos iniciales faltantes se excluyen antes de la partición; no hay imputación con información futura. El escalamiento se aprende dentro del pipeline para evitar fuga. En OLS sin regularización escalar no debe mejorar por sí solo el rendimiento: aporta coherencia al proceso y estabilidad numérica.

Modelo A: β0 + β1 lag_1 + β24 lag_24. Modelo B: añade step y armónicos de 24 y 12 horas. La regla de salida max(0, y_estimado) se define antes de evaluar, porque el objetivo es un conteo no negativo; el modelo ajustado sigue siendo regresión lineal, con posprocesamiento explícito. Se reportan también métricas sin recorte para auditar esta decisión. No se redondean predicciones al calcular errores.

Las transformaciones temporales son deterministas, conocidas anticipadamente. Los datos reales observados de horas previas se permiten en evaluación de una hora. Un horizonte fijo utiliza pronósticos recursivos y jamás inserta los conteos reales futuros.'''),
    ('7. Entrenamiento de modelos', '''Primero se realiza validación temporal expansiva dentro de entrenamiento: tres bloques de 72 horas con train creciente. Los resultados quedan en cv_metrics.csv; sirven para describir estabilidad interna. Los dos candidatos y la regla de recorte son fijos, sin búsqueda de hiperparámetros. Se elige el candidato lineal por MAE en validación: gana lineal_lags (1.938,37 frente a 2.304,44 del temporal). La variante temporal se conserva como contraste requerido y su deterioro se informa explícitamente.

Baselines: media del entrenamiento; persistencia de la hora anterior; repetición de hace 24 horas; repetición de hace 168 horas; promedio de hasta siete observaciones previas de la misma fase diaria. En validación y prueba hay historia suficiente para los siete ciclos. El baseline elegido por MAE de validación es estacional_24 (1.780,71): ya supera al lineal en ese bloque.

Después se reajustan candidatos con train+validación (steps 25-599), se congelan parámetros y se evalúa test una sola vez como protocolo final, sin cambiar selección por lo observado allí. La aplicación reajusta el candidato lineal elegido con las 719 filas, separado del ajuste utilizado para medir test. Se muestran otros candidatos en test como comparación descriptiva, sin reselección.'''),
    ('8. Evaluación del modelo', f'''Protocolo principal: evaluación secuencial de una hora, 144 predicciones en steps 600-743. Se conocen los conteos anteriores hasta t-1; los parámetros no se actualizan dentro del bloque. Para {chosen}: MAE={result.MAE:,.2f}, RMSE={result.RMSE:,.2f}, MAPE={result.MAPE_pct:,.2f}% y R²={result.R2:.4f}. WAPE={result.WAPE_pct:,.2f}%. Las tablas conservan precisión completa.

Frente al baseline estacional_24 seleccionado en validación, el MAE de prueba baja {summary['mejora_MAE_pct_vs_baseline']:.2f}% (1.421,38 a 1.040,06). Sin embargo, persistencia_1 obtiene MAE 1.003,33, menor que el lineal, aunque su RMSE es mayor (1.800,72). No puede afirmarse que la regresión gane a todos los baselines. R² negativo significa SSE mayor que la referencia retrospectiva de la media real del bloque; esa media test no sería conocida para pronosticar.

MAPE=100·media(|y-ŷ|/|y|) solo para y distinto de cero; MAPE_n informa la cantidad válida (144 en test). Aquí no hay ceros, pero hay muchos conteos mínimos, que inflan el porcentaje. MAE mide error absoluto típico en transacciones; RMSE penaliza más los errores grandes. WAPE=100·Σ|error|/Σ|y| complementa MAPE, sin ocultar errores grandes.

Contraste a origen fijo: se predicen todas las 144 horas desde step 599 con rezagos recursivos. El MAE lineal sube a 1.963,35, RMSE a 2.242,74 y R² baja a -1,1207. La repetición recursiva de las últimas 24 horas obtiene MAE 2.047,74. Es una tarea diferente y más difícil; las métricas de una hora no deben publicitarse como rendimiento de 144 horas. No se ofrecen intervalos de incertidumbre sin validación.'''),
    ('9. Conclusiones', '''Se construyó un proceso reproducible de regresión lineal supervisada sobre el mismo objetivo horario de los laboratorios anteriores. Los rezagos de 1 y 24 horas son conocidos a tiempo de predicción y están asociados al objetivo en entrenamiento. Añadir variables temporales conocidas no mejoró el MAE de validación, pese a su significación en el contraste exploratorio; se conserva la variante simple para predicción.

El rendimiento absoluto es insuficiente para uso operativo: R² negativo, MAPE y WAPE muy altos y dependencia del régimen de actividad. La mejora frente al baseline estacional elegido en validación no equivale a superioridad universal. La persistencia de una hora debe mantenerse como referencia fuerte. La selección lineal responde al objetivo académico, no a una afirmación de ser el mejor modelo de producción.

El horizonte recursivo acumula errores y no reemplaza una evaluación específica por horizonte. Próximos pasos: más datos que cubran distintos regímenes, ventanas móviles y detección de cambios, regresión de conteos o transformación del objetivo con corrección adecuada y, como contraste opcional, SARIMA. Estas alternativas requerirían nueva validación temporal antes de cualquier afirmación de mejora.

La app permite filtrar horas, comparar modelos, introducir rezagos manuales y pronosticar de 1 a 72 horas. No se modificó el repositorio remoto. Los resultados anteriores se preservan en references/laboratorio_2; el CSV original no se distribuye porque no está disponible. El proyecto funciona sin él usando la serie agregada de 743 filas.''')]
    cells=[]
    def md(t): cells.append(nbf.v4.new_markdown_cell(t))
    def code(t): cells.append(nbf.v4.new_code_cell(t))
    md('# Laboratorio 3 · PaySim adaptado\nRegresión lineal supervisada del conteo horario\n\n**David Nuncira · Juan Valderrama · Daniel Carrillo**\n\nUPTC · Estadística para Analítica de Datos · 2026-II')
    for i,(title,body) in enumerate(sections,1):
        md('## '+title+'\n\n'+body)
        if i==1:
            code('''# En Colab: subir y extraer laboratorio_3_paysim.zip en /content.
# Si faltan bibliotecas, ejecutar una vez: %pip install -r requirements.txt
from pathlib import Path
import sys, json
candidates = [Path.cwd(), Path.cwd().parent, Path('/content/laboratorio_3_paysim')]
ROOT = next((p for p in candidates if (p/'src/modeling.py').exists()), None)
if ROOT is None: raise FileNotFoundError('Extrae el proyecto completo y abre este notebook desde su carpeta.')
sys.path.insert(0,str(ROOT))
from src.modeling import *
import matplotlib.pyplot as plt
from IPython.display import display, Image
from scripts.run_analysis import run
summary=run()  # Recalcula tablas y gráficos a partir de la serie incluida.
display(pd.Series(summary))''')
        elif i==2: code("previous=json.loads((ROOT/'references/lab2_analysis_summary.json').read_text(encoding='utf-8'))\ndisplay(pd.DataFrame(previous['variable_dictionary']))\ndisplay(pd.DataFrame({'predictor':LAGS+CALENDAR,'disponible_antes_y_t':True}))")
        elif i==3: code("d=load_series(); x=make_features(d)\nassert len(d)==743 and int(d.transacciones_hora.sum())==6362620\nassert len(x)==719\ndisplay(pd.read_csv(ROOT/'results/splits.csv'))\ndisplay(x.head())\ndisplay(previous['data_quality']) # Evidencia previa; no se vuelve a auditar el CSV original.")
        elif i==4: code("display(pd.read_csv(ROOT/'results/descriptive.csv'))\ndisplay(pd.read_csv(ROOT/'results/regime_summary.csv'))\ndisplay(Image(filename=str(ROOT/'results/figures/series.png')))\ndisplay(Image(filename=str(ROOT/'results/figures/eda.png')))")
        elif i==5: code("display(pd.read_csv(ROOT/'results/hypotheses.csv'))\ndisplay(pd.read_csv(ROOT/'results/coefficients_HAC.csv'))\ndisplay(pd.read_csv(ROOT/'results/residual_ljungbox_train.csv'))")
        elif i==6: code("a=int(len(x)*.6); b=int(len(x)*.8)\ntrain,val,test=x.iloc[:a],x.iloc[a:b],x.iloc[b:]\nkind=summary['modelo_seleccionado_validacion_MAE']\nmodel_train=fit_model(train,kind)\nassert np.allclose(model_train['escala'].mean_,train[columns(kind)].mean())\ndisplay(model_train)")
        elif i==7: code("display(pd.read_csv(ROOT/'results/cv_metrics.csv'))\nmetrics=pd.read_csv(ROOT/'results/metrics.csv')\ndisplay(metrics[metrics.etapa=='validacion'])\nmodel_test=fit_model(x.iloc[:b],kind)\nassert x.iloc[:b].step.max()<test.step.min()")
        elif i==8: code("display(metrics[metrics.etapa=='test'])\ndisplay(pd.read_csv(ROOT/'results/recursive_metrics.csv'))\nfor image in ['test.png','residuals.png']: display(Image(filename=str(ROOT/'results/figures'/image)))\n# Contraste independiente con fórmulas explícitas.\ny=test.y.to_numpy(); pred=predict(model_test,test,kind)\nassert np.isclose(np.abs(y-pred).mean(),summary['test_MAE_lineal'])\ndisplay(pd.Series(scores(y,pred)))")
        elif i==9: code("deployment=fit_model(x,kind)\nforecast=recursive(deployment,d,24,kind)\ndisplay(forecast)\nforecast.plot(x='step',y='prediccion',title='Pronóstico recursivo desde step 743'); plt.show()")
    md('### Fuentes y reproducción\n- Guía original: `references/Laboratorio_3.pdf`.\n- [Repositorio Laboratorio 2](https://github.com/david-nuncira/laboratorio_2_paysim): README, serie, resumen y notebooks; huellas en `references/provenance.json`.\n- [Pipeline, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).\n- [TimeSeriesSplit, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).\n- [Covarianza robusta, statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html).\n\nLa evidencia del Laboratorio 1 se limita al esquema y resumen preservados por Laboratorio 2; no se afirma haber revisado un informe adicional del primero.')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':sys.version.split()[0]}})
    (ROOT/'notebooks').mkdir(exist_ok=True)
    if '--pdf-only' not in sys.argv: nbf.write(nb,ROOT/'notebooks/Laboratorio_3.ipynb')
    # Markdown narrativo autónomo.
    report='# Laboratorio 3 · PaySim adaptado\n\nDavid Nuncira · Juan Valderrama · Daniel Carrillo\n\n'+ '\n\n'.join('## '+t+'\n\n'+b for t,b in sections)
    (ROOT/'reports').mkdir(exist_ok=True); (ROOT/'reports/Informe.md').write_text(report,encoding='utf-8')
    # PDF: secciones cortas y saltos controlados para lectura y revisión.
    styles=getSampleStyleSheet(); styles['BodyText'].fontSize=10; styles['BodyText'].leading=14
    story=[]
    def para(t,style='BodyText'):
        for old,new in [('ŷ','y_pred'),('Σ','suma'),('β','beta'),('π','pi'),('α','alpha'),('≈','~')]: t=t.replace(old,new)
        return Paragraph(html.escape(t).replace('\n','<br/>'),styles[style])
    def table(frame):
        f=frame.copy()
        for c in f.select_dtypes('number'):
            f[c]=f[c].map(lambda v: ('< 1e-300' if v==0 else f'{v:.3e}') if c.startswith('p_') else (f'{v:,.4f}' if isinstance(v,float) else str(v)))
        rows=[[para(str(v)) for v in f.columns]]+[[para(str(v)) for v in row] for row in f.itertuples(index=False,name=None)]
        t=Table(rows,repeatRows=1,hAlign='LEFT'); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d9eaf0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#176b87')),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6)]));return t
    story.extend([para('ESTADÍSTICA PARA ANALÍTICA DE DATOS','Heading2'),para('Laboratorio 3','Title'),para('PaySim adaptado: regresión lineal supervisada','Heading1'),para('David Nuncira · Juan Valderrama · Daniel Carrillo'),para('UPTC · Profesor Duván Cataño · 2026-II'),Spacer(1,1*cm),para('Resultados calculados sobre 743 horas y 6.362.620 transacciones agregadas. Entrega ejecutable: notebook, aplicación Streamlit y resultados reproducibles.'),Spacer(1,.5*cm),para('Hallazgo principal','Heading2'),para('La regresión mejora frente al baseline estacional elegido en validación, pero presenta R² negativo y no supera la persistencia en MAE. El desempeño debe interpretarse con cautela ante el cambio de nivel de la serie.'),Image(str(r/'figures/series.png'),width=17*cm,height=6.12*cm),PageBreak()])
    for i,(title,body) in enumerate(sections,1):
        story.append(para(title,'Heading1'))
        for p in body.split('\n\n'): story.extend([para(p),Spacer(1,.22*cm)])
        if i==3: story.append(table(pd.read_csv(r/'splits.csv')))
        if i==4: story.append(Image(str(r/'figures/eda.png'),width=17*cm,height=6.12*cm))
        if i==5: story.append(table(pd.read_csv(r/'hypotheses.csv')[['hipotesis','estadistico','p_Holm','rechaza_005']]))
        if i==7: story.append(table(metrics[(metrics.etapa=='validacion') & metrics.modelo.isin(['lineal_lags','lineal_temporal','estacional_24'])][['modelo','MAE','RMSE','R2']]))
        if i==8:
            story.append(table(test[~test.modelo.str.endswith('sin_recorte')][['modelo','MAE','RMSE','MAPE_pct','R2']]))
            story.append(Image(str(r/'figures/test.png'),width=17*cm,height=6.12*cm))
        if i<9: story.append(PageBreak())
    story.extend([Spacer(1,.4*cm),para('Fuentes y límites documentales','Heading2'),para('Guía docente: references/Laboratorio_3.pdf. Fuente PaySim: https://github.com/david-nuncira/laboratorio_2_paysim. Biblioteca de modelado: documentación oficial de scikit-learn (Pipeline y TimeSeriesSplit) y statsmodels (OLS y covarianza HAC). La trazabilidad y las huellas están en references/provenance.json. Los resultados originales del Laboratorio 2 se conservan sin sobrescribir. No se recuperó el CSV transaccional ni un informe adicional del Laboratorio 1.')])
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#64748b')); canvas.drawString(1.8*cm,1.1*cm,'PaySim | Laboratorio 3 | David Nuncira, Juan Valderrama y Daniel Carrillo'); canvas.drawRightString(19.2*cm,1.1*cm,str(doc.page))
    SimpleDocTemplate(str(ROOT/'reports/Informe_Laboratorio_3.pdf'),pagesize=(21*cm,29.7*cm),rightMargin=1.8*cm,leftMargin=1.8*cm,topMargin=1.7*cm,bottomMargin=1.8*cm).build(story,onFirstPage=footer,onLaterPages=footer)
    print('Notebook, narrativa y PDF generados')

if __name__=='__main__': build()
