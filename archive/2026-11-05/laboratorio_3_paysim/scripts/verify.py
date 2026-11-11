from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.modeling import *

def verify():
    checks={}; d=load_series(); x=make_features(d); a=int(.6*len(x)); b=int(.8*len(x))
    assert len(d)==743 and d.transacciones_hora.sum()==6362620
    checks['serie_continua_total_y_filas']=True
    # Auditoría de disponibilidad: todos los lags se comparan con el dato histórico exacto.
    lookup=d.set_index('step').transacciones_hora
    for lag in [1,24]:
        np.testing.assert_array_equal(x[f'lag_{lag}'],lookup.loc[x.step-lag].to_numpy())
    changed=d.copy(); changed.loc[changed.step>=600,'transacciones_hora']+=999999
    fx=make_features(changed)
    np.testing.assert_array_equal(x[x.step<=600][LAGS].to_numpy(),fx[fx.step<=600][LAGS].to_numpy())
    checks['lags_no_usan_objetivo_actual_ni_futuro']=True
    m=fit_model(x.iloc[:a],'lineal_temporal')
    np.testing.assert_allclose(m['escala'].mean_,x.iloc[:a][columns('lineal_temporal')].mean())
    checks['escalador_ajustado_solo_train']=True
    s=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8')); kind=s['modelo_seleccionado_validacion_MAE']
    model=fit_model(x.iloc[:b],kind); y=x.iloc[b:].y.to_numpy(); p=predict(model,x.iloc[b:],kind)
    row=pd.read_csv(ROOT/'results/metrics.csv').query('etapa=="test" and modelo==@kind').iloc[0]
    np.testing.assert_allclose([row.MAE,row.RMSE,row.MAPE_pct,row.R2],
        [np.abs(y-p).mean(),np.sqrt(np.mean((y-p)**2)),100*np.mean(np.abs((y-p)/y)),1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)])
    checks['metricas_con_formulas_independientes']=True
    rec=recursive(model,d[d.step<600],144,kind)
    np.testing.assert_allclose(rec.prediccion,pd.read_csv(ROOT/'results/recursive_test.csv').prediccion)
    checks['recursivo_sin_observaciones_test']=True
    # Verificar que las copias locales de notebooks coinciden con los blobs remotos.
    prov=json.loads((ROOT/'references/provenance.json').read_text(encoding='utf-8'))
    for entry in prov['files']:
        if not entry['path'].startswith('notebooks/'): continue
        content=(ROOT/'references/laboratorio_2'/entry['path']).read_bytes()
        blob=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
        if blob!=entry['github_blob_sha']: raise AssertionError(f'Copia distinta del notebook GitHub: {entry["path"]}')
    checks['notebooks_previos_identicos_a_GitHub']=True
    # Pruebas funcionales de pantallas y controles reales del dashboard.
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert not app.exception
    pages=['Proyecto','Exploración','Evaluación','Pronosticar','Descargas']
    for page in pages:
        app.sidebar.radio[0].set_value(page).run(); assert not app.exception, str(app.exception)
        if page=='Exploración':
            app.slider[0].set_value((500,743)).run(); assert not app.exception
        if page=='Evaluación':
            app.selectbox[0].set_value('validacion').run(); assert not app.exception
            app.multiselect[0].set_value(['lineal_lags']).run(); assert not app.exception
        if page=='Pronosticar':
            app.number_input[1].set_value(12); app.button[0].click().run(); assert not app.exception
            assert len(app.metric)>0
            app.radio[0].set_value('Próximas horas: recursivo').run(); assert not app.exception
            app.slider[0].set_value(48).run(); assert not app.exception
    checks['streamlit_5_pantallas_filtros_prediccion_y_horizonte']=True
    (ROOT/'results/verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__': verify()
