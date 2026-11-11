"""Opcional: reconstruye la serie si se dispone del CSV original, sin sobrescribir fuentes."""
from pathlib import Path
import argparse,hashlib
import pandas as pd
parser=argparse.ArgumentParser()
parser.add_argument('--data',required=True,help='Ruta al CSV transaccional original adaptado')
parser.add_argument('--output',default='results/hourly_rebuilt.csv')
args=parser.parse_args()
source=Path(args.data); target=Path(args.output)
if target.exists(): raise FileExistsError('Para preservar resultados, usa una salida nueva.')
digest=hashlib.sha256()
with source.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
print('SHA256 original:',digest.hexdigest())
# Solo step es necesario para validar el objetivo; esto evita cargar 385 MB en RAM.
counts={}
for chunk in pd.read_csv(source,usecols=['step'],chunksize=250000):
    v=pd.to_numeric(chunk.step,errors='raise')
    if v.isna().any() or not (v%1==0).all(): raise ValueError('step no entero o ausente')
    for step,count in v.value_counts().items(): counts[int(step)]=counts.get(int(step),0)+int(count)
out=pd.DataFrame(sorted(counts.items()),columns=['step','transacciones_hora'])
target.parent.mkdir(parents=True,exist_ok=True); out.to_csv(target,index=False)
print('Conteos guardados sin reemplazar hourly_series.csv:',target)
