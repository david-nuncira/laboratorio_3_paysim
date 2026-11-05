from pathlib import Path
import sys, json, hashlib, platform, importlib.metadata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import TimeSeriesSplit
from src.modeling import *

def run():
    dest=ROOT/'results'; dest.mkdir(exist_ok=True)
    figdir=dest/'figures'; figdir.mkdir(exist_ok=True)
    d=load_series(); x=make_features(d)
    n=len(x); a=int(n*.6); b=int(n*.8)
    train,val,test=x.iloc[:a],x.iloc[a:b],x.iloc[b:]
    split=pd.DataFrame([{'grupo':name,'n':len(v),'inicio':int(v.step.min()),'fin':int(v.step.max())} for name,v in [('train',train),('validacion',val),('test',test)]]); split.to_csv(dest/'splits.csv',index=False)
    x.to_csv(dest/'supervised.csv',index=False)
    d.describe().to_csv(dest/'descriptive.csv')
    d.assign(bloque=((d.step-1)//168)+1).groupby('bloque').transacciones_hora.agg(['count','mean','median','min','max']).to_csv(dest/'regime_summary.csv')
    rows=[]; predictions=[]; cv=[]
    kinds=['lineal_lags','lineal_temporal']
    # CV solo dentro del primer bloque; el test no participa en selección.
    for fold,(tr,va) in enumerate(TimeSeriesSplit(n_splits=3,test_size=72).split(train),1):
        for kind in kinds:
            m=fit_model(train.iloc[tr],kind)
            cv.append({'fold':fold,'modelo':kind,'train_fin':int(train.iloc[tr].step.max()),'val_inicio':int(train.iloc[va].step.min()),'val_fin':int(train.iloc[va].step.max()),**scores(train.iloc[va].y,predict(m,train.iloc[va],kind))})
    pd.DataFrame(cv).to_csv(dest/'cv_metrics.csv',index=False)
    # Modelos congelados durante cada bloque. Actualización de observaciones, no de parámetros.
    for stage,fit,target in [('validacion',train,val),('test',x.iloc[:b],test)]:
        pred=baselines(fit,target)
        for kind in kinds:
            m=fit_model(fit,kind); pred[kind]=predict(m,target,kind)
            raw=m.predict(target[columns(kind)])
            rows.append({'etapa':stage,'modelo':kind+'_sin_recorte',**scores(target.y,raw)})
        for name,p in pred.items():
            if not np.isfinite(p).all(): raise ValueError('Baseline sin historia suficiente')
            rows.append({'etapa':stage,'modelo':name,**scores(target.y,p)})
            predictions.extend([{'etapa':stage,'modelo':name,'step':int(s),'real':float(y),'prediccion':float(v)} for s,y,v in zip(target.step,target.y,p)])
    metrics=pd.DataFrame(rows); metrics.to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(dest/'predictions.csv',index=False)
    eligible=metrics[(metrics.etapa=='validacion')&metrics.modelo.isin(kinds)]
    selected=eligible.sort_values('MAE').iloc[0].modelo
    baseline=metrics[(metrics.etapa=='validacion')&~metrics.modelo.str.startswith('lineal')].sort_values('MAE').iloc[0].modelo
    # Inferencia exploratoria solo entrenamiento. HAC considera autocorrelación hasta 24 horas.
    cols=columns('lineal_temporal')
    X=sm.add_constant(train[cols])
    ols=sm.OLS(train.y,X).fit(cov_type='HAC',cov_kwds={'maxlags':24,'use_correction':True})
    hypotheses=[]
    for group,terms in [('rezagos',LAGS),('temporales',CALENDAR)]:
        R=np.zeros((len(terms),len(X.columns)))
        for i,t in enumerate(terms): R[i,list(X.columns).index(t)]=1
        w=ols.wald_test(R,scalar=True)
        hypotheses.append({'hipotesis':group,'H0':'Todos los coeficientes del grupo son cero','estadistico':float(w.statistic),'p_HAC':float(w.pvalue)})
    reject,padj,_,_=multipletests([r['p_HAC'] for r in hypotheses],alpha=.05,method='holm')
    for row,r,p in zip(hypotheses,reject,padj): row.update(p_Holm=float(p),rechaza_005=bool(r))
    pd.DataFrame(hypotheses).to_csv(dest/'hypotheses.csv',index=False)
    ci=ols.conf_int(); pd.DataFrame({'variable':X.columns,'coef':ols.params,'se_HAC':ols.bse,'p_HAC':ols.pvalues,'IC95_inf':ci[0],'IC95_sup':ci[1]}).to_csv(dest/'coefficients_HAC.csv',index=False)
    acorr_ljungbox(ols.resid,lags=[24],return_df=True).to_csv(dest/'residual_ljungbox_train.csv')
    # Origen fijo: ningún conteo real del bloque test entra en los predictores.
    pretest=d[d.step<test.step.min()]; mtest=fit_model(x.iloc[:b],selected)
    rec=recursive(mtest,pretest,len(test),selected); rec['real']=test.y.to_numpy()
    rec.to_csv(dest/'recursive_test.csv',index=False)
    pd.DataFrame([{'modelo':selected,'protocolo':'origen_fijo_test',**scores(rec.real,rec.prediccion)},
                  {'modelo':'estacional_24','protocolo':'origen_fijo_test',**scores(test.y,np.resize(pretest.transacciones_hora.iloc[-24:].to_numpy(),len(test)))}]).to_csv(dest/'recursive_metrics.csv',index=False)
    # Artefacto de despliegue reajustado con toda la historia; distinto del modelo de evaluación.
    deployment=fit_model(x,selected)
    recursive(deployment,d,24,selected).to_csv(dest/'forecast_next24.csv',index=False)
    tmetric=metrics[(metrics.etapa=='test')&(metrics.modelo==selected)].iloc[0]
    bmetric=metrics[(metrics.etapa=='test')&(metrics.modelo==baseline)].iloc[0]
    summary={'n_transacciones':int(d.transacciones_hora.sum()),'n_horas':len(d),'n_supervisadas':n,'modelo_seleccionado_validacion_MAE':selected,'baseline_seleccionado_validacion_MAE':baseline,
             'test_MAE_lineal':float(tmetric.MAE),'test_MAE_baseline':float(bmetric.MAE),'mejora_MAE_pct_vs_baseline':float(100*(bmetric.MAE-tmetric.MAE)/bmetric.MAE),
             'sha256_hourly':hashlib.sha256((dest/'hourly_series.csv').read_bytes()).hexdigest(),
             'protocol':'rolling one-step, parámetros congelados, lags reales previos disponibles; recorte a cero predefinido',
             'python':platform.python_version(),'versions':{p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','statsmodels','streamlit']}}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    plt.rcParams.update({'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,ax=plt.subplots(figsize=(10,3.6)); ax.plot(d.step,d.transacciones_hora,lw=1,color='#176b87'); ax.axvline(train.step.max(),color='orange',ls='--',label='Fin entrenamiento'); ax.axvline(val.step.max(),color='red',ls='--',label='Fin validación'); ax.set(xlabel='Hora simulada (step)',ylabel='Transacciones',title='Volumen horario y partición cronológica'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'series.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6)); axes[0].hist(train.y,bins=35,color='#176b87'); axes[0].set(title='Distribución del objetivo en train',xlabel='Transacciones'); train.groupby(train.step%24).y.mean().plot(ax=axes[1],marker='o',color='#176b87'); axes[1].set(title='Perfil por fase de 24 h (train)',xlabel='step % 24',ylabel='Conteo medio'); fig.tight_layout(); fig.savefig(figdir/'eda.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.6)); ax.plot(test.step,test.y,label='Real',color='#17324d'); ax.plot(test.step,predict(mtest,test,selected),label=selected,alpha=.8); ax.plot(test.step,baselines(x.iloc[:b],test)[baseline],label=baseline,alpha=.6); ax.set(xlabel='Hora simulada',ylabel='Transacciones',title='Prueba: pronósticos de una hora'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'test.png'); plt.close(fig)
    errors=test.y.to_numpy()-predict(mtest,test,selected)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6)); axes[0].plot(test.step,errors,lw=1); axes[0].axhline(0,color='grey'); axes[0].set(title='Error en prueba: real - predicción',xlabel='step'); axes[1].scatter(predict(mtest,test,selected),errors,s=10); axes[1].axhline(0,color='grey'); axes[1].set(xlabel='Predicción',ylabel='Error',title='Dispersión residual'); fig.tight_layout(); fig.savefig(figdir/'residuals.png'); plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary

if __name__=='__main__': run()
