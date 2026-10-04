@echo off
rem Daily live nowcast update, for Windows Task Scheduler (docs/runbook.md, "Live mode").
rem Runs `oceanembed live update` from the project root and writes a log per run to
rem outputs\live\logs\update_<date>_<time>.log. Exit code = that of the update (0 = ok).
rem Safe to run by hand and safe to run twice: an update that finds nothing new changes nothing.
setlocal enableextensions
cd /d "%~dp0.."
if not exist "outputs\live\logs" mkdir "outputs\live\logs"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
set "LOG=outputs\live\logs\update_%STAMP%.log"
echo [%date% %time%] live update started > "%LOG%"
".venv\Scripts\oceanembed.exe" live update --config configs\live.yaml >> "%LOG%" 2>&1
set "RC=%errorlevel%"
echo [%date% %time%] finished with exit code %RC% >> "%LOG%"
exit /b %RC%
