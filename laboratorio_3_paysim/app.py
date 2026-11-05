from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.modeling import ROOT, load_series, make_features, fit_model, predict, recursive

st.set_page_config(page_title='PaySim | Laboratorio 3',page_icon='📈',layout='wide')
@st.cache_data
def data():
    return load_series(),json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
d,s=data(); x=make_features(d); kind=s['modelo_seleccionado_validacion_MAE']
st.title('PaySim · Pronóstico de transacciones')
st.caption('Laboratorio 3 · David Nuncira, Juan Valderrama y Daniel Carrillo · UPTC')
page=st.sidebar.radio('Sección',['Proyecto','Exploración','Evaluación','Pronosticar','Descargas'])
st.sidebar.caption('Fuente: Laboratorio 2. Tiempo simulado; no son fechas civiles ni operaciones reales.')
if page=='Proyecto':
    st.header('De la estadística descriptiva a la predicción')
    st.write('Objetivo: estimar cuántas transacciones ocurrirán en la próxima hora simulada mediante regresión lineal supervisada. Se utilizan conteos de hace 1 y 24 horas y, en la variante temporal, senos/cosenos de fase y step. La variante se elige solo por MAE de validación.')
    c=st.columns(3)
    for col,label,value in zip(c,['Transacciones','Horas','Variables originales'],[f'{s["n_transacciones"]:,}',s['n_horas'],7]): col.metric(label,value)
    st.info('El CSV transaccional original no está incluido ni se recuperó en esta sesión. Los conteos horarios se recuperaron de GitHub; su suma coincide con 6.362.620. La limpieza de las 7 columnas originales se conserva como resultado previo, sin afirmar que se recalculó.')
    st.write('Modelo seleccionado:',kind)
    st.write('El objetivo es un conteo numérico: corresponde regresión lineal, no clasificación logística. No se usan montos contemporáneos ni cuentas como predictores.')
    st.dataframe(pd.read_csv(ROOT/'results/splits.csv'),hide_index=True)
elif page=='Exploración':
    lo,hi=st.slider('Intervalo de horas',1,743,(1,743))
    v=d[d.step.between(lo,hi)]
    st.plotly_chart(px.line(v,x='step',y='transacciones_hora',title='Conteo horario'),width='stretch')
    st.plotly_chart(px.box(v.assign(fase=(v.step%24).astype(str)),x='fase',y='transacciones_hora',title='Distribución por fase de 24 horas'),width='stretch')
    st.dataframe(v.describe())
    st.caption('Exploración interactiva de toda la serie; las decisiones del modelo se limitaron a entrenamiento y validación. Los picos se conservan como observaciones válidas.')
elif page=='Evaluación':
    stage=st.selectbox('Bloque',['test','validacion'])
    metrics=pd.read_csv(ROOT/'results/metrics.csv'); st.dataframe(metrics[metrics.etapa==stage],hide_index=True)
    ps=pd.read_csv(ROOT/'results/predictions.csv'); ps=ps[ps.etapa==stage]
    models=st.multiselect('Modelos',list(ps.modelo.unique()),default=[kind,s['baseline_seleccionado_validacion_MAE']])
    if models:
        chart=ps[ps.modelo.isin(models)].pivot(index='step',columns='modelo',values='prediccion')
        chart['real']=ps.groupby('step').real.first()
        st.plotly_chart(px.line(chart,title='Evaluación secuencial de una hora'),width='stretch')
    st.warning('Cada pronóstico de una hora usa observaciones anteriores del bloque ya conocidas. Esto no equivale a pronosticar 144 horas desde un único origen. Los parámetros permanecen fijos en cada bloque.')
    st.subheader('Contraste desde un único origen')
    st.dataframe(pd.read_csv(ROOT/'results/recursive_metrics.csv'),hide_index=True)
    st.caption('MAPE en porcentaje, excluyendo solo objetivos cero (cantidad válida en MAPE_n). Conteos pequeños pueden inflar MAPE. R² negativo indica peor ajuste que la media del bloque evaluado.')
elif page=='Pronosticar':
    mode=st.radio('Modo',['Próxima hora: escenario manual','Próximas horas: recursivo'])
    # Reajuste con toda la historia después de evaluar; no se usa este ajuste en métricas test.
    model=fit_model(x,kind)
    if mode=='Próxima hora: escenario manual':
        st.write('Introduce conteos realmente disponibles antes de la hora objetivo. Los valores iniciales corresponden al step 744.')
        step=st.number_input('Hora objetivo (step)',min_value=25,value=744,step=1)
        lag1=st.number_input('Conteo de la hora anterior (lag 1)',min_value=0,value=int(d.transacciones_hora.iloc[-1]),step=1)
        lag24=st.number_input('Conteo de hace 24 horas (lag 24)',min_value=0,value=int(d.transacciones_hora.iloc[-24]),step=1)
        if st.button('Calcular pronóstico',type='primary'):
            row={'step':step,'lag_1':lag1,'lag_24':lag24}
            for p in [24,12]:
                row[f'sin{p}']=np.sin(2*np.pi*step/p); row[f'cos{p}']=np.cos(2*np.pi*step/p)
            raw=float(model.predict(pd.DataFrame([row])[model.feature_names_in_])[0]); value=float(predict(model,pd.DataFrame([row]),kind)[0])
            st.metric('Transacciones estimadas',f'{value:,.2f}')
            st.caption(f'Salida lineal sin recorte: {raw:,.2f}. Regla operacional: máximo entre cero y la salida; no se redondea para calcular métricas.')
            if step>767 or max(lag1,lag24)>d.transacciones_hora.max(): st.warning('Escenario fuera del rango reciente observado; la extrapolación es incierta.')
    else:
        horizon=st.slider('Horizonte en horas',1,72,24)
        fc=recursive(model,d,horizon,kind)
        st.plotly_chart(px.line(fc,x='step',y='prediccion',markers=True,title='Pronóstico desde la última hora observada'),width='stretch')
        st.dataframe(fc,hide_index=True)
        st.download_button('Descargar pronóstico CSV',fc.to_csv(index=False).encode('utf-8'),'pronostico.csv','text/csv')
        st.warning('A partir de la segunda hora, lag 1 contiene pronósticos. A partir de la hora 25 también lag 24. El error puede acumularse; no se presenta un intervalo de confianza no validado.')
    st.caption('Modelo de uso reajustado con todas las horas disponibles. Las métricas de prueba pertenecen a un ajuste anterior. Las predicciones son del simulador y no recomendaciones financieras.')
else:
    for filename in ['metrics.csv','predictions.csv','hypotheses.csv','cv_metrics.csv','forecast_next24.csv','summary.json','recursive_metrics.csv']:
        st.download_button(filename,(ROOT/'results'/filename).read_bytes(),filename)
