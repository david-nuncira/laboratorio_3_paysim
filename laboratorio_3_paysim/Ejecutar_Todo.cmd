@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( python -m venv .venv )
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto error
".venv\Scripts\python.exe" scripts\verify.py || goto error
".venv\Scripts\python.exe" scripts\run_analysis.py || goto error
".venv\Scripts\python.exe" scripts\build_report.py || goto error
".venv\Scripts\python.exe" scripts\build_notebook.py || goto error
".venv\Scripts\python.exe" scripts\execute_notebook.py || goto error
echo.
echo Listo: results\, reports\Informe_Laboratorio_3.md y notebooks\Laboratorio_3.ipynb actualizados.
echo Para el dashboard ejecuta Iniciar_Streamlit.cmd
pause
exit /b 0
:error
echo Fallo en uno de los pasos. Revisa el mensaje anterior.
pause
exit /b 1
