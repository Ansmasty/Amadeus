@echo off
chcp 65001 >nul
title AMADEUS - Micrófonos disponibles
color 0F

echo.
python main.py --list-mics
echo.
pause