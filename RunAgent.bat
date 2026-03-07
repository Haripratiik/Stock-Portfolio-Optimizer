@echo off
title Portfolio Pipeline Agent
cd /d "%~dp0"

REM ── Activate venv if present ──
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo Starting Local Agent...
python backend\LocalAgent.py

echo.
echo Agent closed.
pause
