@echo off
setlocal
cd /d "%~dp0"
call .venv310\Scripts\activate
python -X utf8 main.py --voice
pause