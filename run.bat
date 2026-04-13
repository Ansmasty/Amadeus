@echo off
setlocal
cd /d "%~dp0"

if not exist logs mkdir logs
set LOG=logs\run_%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG=%LOG: =0%

chcp 65001 >nul
title AMADEUS - TEXTO
color 0A

echo Iniciando AMADEUS (texto)...
where py >nul 2>&1
if %errorlevel%==0 (
    py -3.10 -X utf8 main.py 1>"%LOG%" 2>&1
) else (
    python -X utf8 main.py 1>"%LOG%" 2>&1
)

echo.
echo Exit code: %errorlevel%
echo Log: "%LOG%"
type "%LOG%"
pause