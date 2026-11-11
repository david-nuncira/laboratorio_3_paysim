@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creando entorno de Python...
  python -m venv .venv
  if errorlevel 1 goto error
)
".venv\Scripts\python.exe" -c "import numpy, pandas, scipy, sklearn, plotly, streamlit" >nul 2>&1
if errorlevel 1 (
  echo Instalando bibliotecas del proyecto...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto error
)
if not exist "results\hourly_series.csv" if not exist "data\hourly_series.csv" (
  echo.
  echo FALTA LA SERIE: copia hourly_series.csv del Laboratorio 2 en la carpeta results\ y vuelve a ejecutar.
  goto error
)
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto error
exit /b 0
:error
echo No se pudo iniciar. Revisa el mensaje anterior.
pause
exit /b 1
