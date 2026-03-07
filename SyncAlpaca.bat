@echo off
cd /d "%~dp0"
python backend/sync_alpaca.py
pause
