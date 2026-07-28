@echo off
setlocal
cd /d "%~dp0"
if exist .venv310\Scripts\activate.bat (
	call .venv310\Scripts\activate.bat
)

where py >nul 2>&1
if %errorlevel%==0 (
	py -3.10 -X utf8 main.py --voice
) else (
	python -X utf8 main.py --voice
)
pause