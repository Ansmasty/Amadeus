@echo off
setlocal
cd /d "%~dp0"

if not exist logs mkdir logs
set LOG=logs\run_gui_%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG=%LOG: =0%

chcp 65001 >nul
title AMADEUS - GUI
color 0B

if not exist ".env" (
    echo [AVISO] No se encontro el archivo .env en el directorio del proyecto.
)

if exist ".venv310\Scripts\activate.bat" (
    call ".venv310\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

echo Iniciando AMADEUS (interfaz grafica)...
where py >nul 2>&1
if %errorlevel%==0 (
    py -3.13 -X utf8 gui.py %* 1>"%LOG%" 2>&1
) else (
    python -X utf8 gui.py %* 1>"%LOG%" 2>&1
)

echo.
echo Exit code: %errorlevel%
echo Log: "%LOG%"
type "%LOG%"
pause
