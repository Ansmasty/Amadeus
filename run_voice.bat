@echo off
setlocal
cd /d "%~dp0"
if exist .venv310\Scripts\activate.bat (
	call .venv310\Scripts\activate.bat
)

py -3.10 -X utf8 main.py --voice
pause