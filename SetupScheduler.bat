@echo off
:: SetupScheduler.bat
:: ==================
:: Registers a Windows Task Scheduler task that runs SchedulerCron.py
:: every hour, even when the Local Agent GUI is closed.
::
:: Run this ONCE (as Administrator for best results, but not required).
:: To unregister:  schtasks /Delete /TN "PortfolioManagerScheduler" /F

title Portfolio Manager — Scheduler Setup
cd /d "%~dp0"

:: Resolve paths
set "PROJECT_DIR=%~dp0"
set "PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
set "SCRIPT=%PROJECT_DIR%backend\SchedulerCron.py"
set "TASK_NAME=PortfolioManagerScheduler"
set "LOG_DIR=%PROJECT_DIR%logs"

:: Fallback to system python if venv not found
if not exist "%PYTHON%" (
    echo [Setup] .venv not found, using system python...
    set "PYTHON=python"
)

:: Create logs directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.
echo  Portfolio Manager — Scheduler Setup
echo  =====================================
echo  Project : %PROJECT_DIR%
echo  Python  : %PYTHON%
echo  Script  : %SCRIPT%
echo  Task    : %TASK_NAME%
echo  Runs    : Every 1 hour
echo.

:: Delete existing task if present (so we can update it cleanly)
schtasks /Query /TN "%TASK_NAME%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [Setup] Removing existing task...
    schtasks /Delete /TN "%TASK_NAME%" /F >nul
)

:: Register the hourly task
:: /SC HOURLY          — run every hour
:: /MO 1              — modifier: every 1 hour
:: /ST 00:00          — start time (midnight, then every hour)
:: /RL HIGHEST         — run with highest available privileges
:: /RU ""              — run as current user
:: /F                  — force create
schtasks /Create ^
    /TN "%TASK_NAME%" ^
    /TR "\"%PYTHON%\" \"%SCRIPT%\"" ^
    /SC HOURLY ^
    /MO 1 ^
    /ST 00:00 ^
    /RL HIGHEST ^
    /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo  [OK] Task registered successfully.
    echo.
    echo  The scheduler will run SchedulerCron.py every hour automatically.
    echo  It queues overdue pipeline tasks to Firestore even when the
    echo  Local Agent GUI is closed.
    echo.
    echo  To verify:   schtasks /Query /TN "%TASK_NAME%" /V /FO LIST
    echo  To run now:  schtasks /Run /TN "%TASK_NAME%"
    echo  To remove:   schtasks /Delete /TN "%TASK_NAME%" /F
    echo.
) else (
    echo.
    echo  [ERROR] Failed to register task (exit code %ERRORLEVEL%).
    echo  Try running this bat file as Administrator.
    echo.
    echo  Manual alternative — run this command in an admin PowerShell:
    echo    schtasks /Create /TN "%TASK_NAME%" /TR "'%PYTHON%' '%SCRIPT%'" /SC HOURLY /MO 1 /F
    echo.
)

pause
