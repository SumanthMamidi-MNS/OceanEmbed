@echo off
rem OceanEmbed one-click launcher: checks, sets up the project's own environment, starts the
rem dashboard (oceanembed serve) on a free port and opens it in the default browser.
rem Safe to run again. Installs nothing outside this folder. Set OCEANEMBED_NO_BROWSER=1 to skip
rem opening the browser, OCEANEMBED_PORT=<n> to ask for another first port (default 8000).
setlocal enableextensions
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"
set "OE=%CD%\.venv\Scripts\oceanembed.exe"

echo.
echo === OceanEmbed ===

rem --- 1. Python environment (.venv in this folder) --------------------------------------
if not exist "%PY%" (
  py -3.12 -c "import sys" >nul 2>&1
  if errorlevel 1 (
    echo [missing] Python 3.12 with the "py" launcher was not found.
    echo           Install it from https://www.python.org/downloads/ ^(tick "py launcher"^), then run start.bat again.
    exit /b 1
  )
  echo [setup] creating .venv with Python 3.12 ...
  py -3.12 -m venv .venv || goto :fail
  "%PY%" -m pip install --upgrade pip --quiet || goto :fail
)
echo [ok] Python environment: .venv

rem --- 2. PyTorch (CUDA build first when an NVIDIA GPU is present) ---------------------------
"%PY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
  nvidia-smi -L >nul 2>&1
  if errorlevel 1 (
    echo [setup] no NVIDIA GPU found: installing the CPU build of PyTorch ^(a few minutes^) ...
    "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet || goto :fail
  ) else (
    echo [setup] NVIDIA GPU found: installing the CUDA build of PyTorch ^(a few minutes^) ...
    "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --quiet || goto :fail
  )
)
echo [ok] PyTorch

rem --- 3. Project dependencies -------------------------------------------------------------
"%PY%" -c "import oceanembed, fastapi, uvicorn, lightgbm, xarray, zarr" >nul 2>&1
if errorlevel 1 (
  echo [setup] installing the project and its dependencies ^(pip install -e .^) ...
  "%PY%" -m pip install -e . --quiet || goto :fail
)
echo [ok] oceanembed and its dependencies

rem --- 4. Web UI (web\dist) ------------------------------------------------------------------
if not exist "web\dist\index.html" (
  where npm >nul 2>&1
  if errorlevel 1 (
    echo [missing] web\dist is not built and Node.js / npm was not found.
    echo           Install Node.js 20 or newer from https://nodejs.org/ then run start.bat again.
    exit /b 1
  )
  echo [setup] building the web UI ^(npm install, npm run build^) ...
  pushd web
  if not exist node_modules call npm install --no-audit --no-fund || (popd & goto :fail)
  call npm run build || (popd & goto :fail)
  popd
)
echo [ok] web UI: web\dist

rem --- 5. Assets ------------------------------------------------------------------------------
if exist "models\final\recon.pt" (
  echo [ok] released weights: models\final
) else (
  echo [note] models\final\recon.pt is missing: the dashboard still works, but `oceanembed predict`
  echo        and the live nowcast need the released weights ^(see models\final\MODEL_CARD.md^).
)
if not exist "outputs" mkdir "outputs"
set "HAVE_RUN="
for /d %%D in ("outputs\*") do if exist "%%D\run_meta.json" set "HAVE_RUN=1"
if defined HAVE_RUN (
  echo [ok] finished run^(s^) found in outputs\
) else (
  echo [note] no finished run in outputs\: the page will explain that it has nothing to show yet.
  echo        Produce one with: .venv\Scripts\oceanembed.exe run-all --config configs\synthetic.yaml
)

rem --- 6. Free port, server, browser --------------------------------------------------------
if not defined OCEANEMBED_PORT set "OCEANEMBED_PORT=8000"
set "PORT="
for /f %%P in ('powershell -NoProfile -Command "$p=[int]$env:OCEANEMBED_PORT; while($true){ try{ $l=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$p); $l.Start(); $l.Stop(); break }catch{ $p++ } }; $p"') do set "PORT=%%P"
if not defined PORT (
  echo [error] could not find a free port.
  exit /b 1
)
if not "%OCEANEMBED_NO_BROWSER%"=="1" (
  start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 90;$i++){ try{ Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%PORT%/api/health' -TimeoutSec 2 | Out-Null; Start-Process 'http://127.0.0.1:%PORT%'; break }catch{ Start-Sleep 1 } }"
)
echo.
echo Dashboard: http://127.0.0.1:%PORT%   ^(close this window or press Ctrl+C to stop^)
echo.
"%OE%" serve --port %PORT%
exit /b %errorlevel%

:fail
echo.
echo [error] a setup step failed, see the message above.
exit /b 1
