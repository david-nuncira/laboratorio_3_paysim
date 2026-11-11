"""Regresión supervisada con disponibilidad temporal explícita."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
LAGS = ['lag_1', 'lag_24']
CALENDAR = ['sin24', 'cos24', 'sin12', 'cos12', 'step']

def load_series(path=None):
    d = pd.read_csv(path or ROOT/'results/hourly_series.csv').sort_values('step').reset_index(drop=True)
    for col in ['step', 'transacciones_hora']:
        d[col] = pd.to_numeric(d[col], errors='raise')
        if not np.isfinite(d[col]).all() or not (d[col] % 1 == 0).all():
            raise ValueError(f'{col}: valores no finitos o no enteros')
    if d.step.duplicated().any() or not (d.step.diff().dropna() == 1).all():
        raise ValueError('Se requieren horas únicas y consecutivas; no imputar ausencias como cero.')
    if (d.transacciones_hora < 0).any():
        raise ValueError('Conteos negativos')
    return d

def make_features(d):
    x = pd.DataFrame({'step': d.step, 'y': d.transacciones_hora})
    for lag in [1, 24, 168]:
        x[f'lag_{lag}'] = x.y.shift(lag)
    for period in [24, 12]:
        x[f'sin{period}'] = np.sin(2*np.pi*x.step/period)
        x[f'cos{period}'] = np.cos(2*np.pi*x.step/period)
    x['seasonal_mean_7'] = pd.concat([x.y.shift(24*k) for k in range(1,8)], axis=1).mean(axis=1)
    # Solo lag_1 y lag_24 son predictores del modelo. Lag168 sirve de baseline.
    return x.dropna(subset=LAGS).reset_index(drop=True)

def columns(kind):
    return LAGS if kind == 'lineal_lags' else LAGS + CALENDAR

def pipeline():
    return Pipeline([('escala', StandardScaler()), ('regresion', LinearRegression())])

def fit_model(train, kind):
    return pipeline().fit(train[columns(kind)], train.y)

def predict(model, d, kind):
    return np.maximum(0, model.predict(d[columns(kind)]))

def scores(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    nonzero = y != 0
    return {'MAE': float(mean_absolute_error(y,pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y,pred))),
            'MAPE_pct': float(100*np.mean(np.abs((y[nonzero]-pred[nonzero])/y[nonzero]))) if nonzero.any() else None,
            'MAPE_n': int(nonzero.sum()), 'R2': float(r2_score(y,pred)),
            'WAPE_pct': float(100*np.abs(y-pred).sum()/np.abs(y).sum()) if np.abs(y).sum() else None}

def baselines(train, target):
    return {'media_train': np.repeat(train.y.mean(),len(target)),
            'persistencia_1': target.lag_1.to_numpy(),
            'estacional_24': target.lag_24.to_numpy(),
            'estacional_168': target.lag_168.to_numpy(),
            'media_estacional_7': target.seasonal_mean_7.to_numpy()}

def recursive(model, history, horizon, kind):
    hist = history[['step','transacciones_hora']].copy()
    out = []
    for _ in range(horizon):
        step = int(hist.step.iloc[-1])+1
        row = {'step': step, 'lag_1': float(hist.transacciones_hora.iloc[-1]),
               'lag_24': float(hist.transacciones_hora.iloc[-24])}
        for p in [24,12]:
            row[f'sin{p}'] = np.sin(2*np.pi*step/p)
            row[f'cos{p}'] = np.cos(2*np.pi*step/p)
        value = float(predict(model,pd.DataFrame([row]),kind)[0])
        out.append({'step':step,'prediccion':value})
        hist.loc[len(hist)] = [step,value]
    return pd.DataFrame(out)
