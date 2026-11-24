@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PIP_NO_CACHE_DIR=1
set TEMP=%~dp0tmp
set TMP=%~dp0tmp
if not exist tmp mkdir tmp

if exist venv\Scripts\python.exe goto pkgs

echo [1/3] Looking for installed Python 3.10 - 3.13 ...
set PY=
for %%C in ("python" "py -3.13" "py -3.12" "py -3.11" "py -3.10") do call :try %%~C
if defined PY goto mkvenv
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PY goto mkvenv
echo No usable Python found. Installing Python 3.12 ...
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

:mkvenv
echo Using Python: %PY%
%PY% -m venv venv
if errorlevel 1 goto fail

:pkgs
if not exist venv\ready.txt goto install
fc /b requirements.txt venv\req.txt >nul 2>&1
if errorlevel 1 goto install
goto run
:install
echo [2/3] Installing packages - first run only, takes a few minutes ...
venv\Scripts\python -m pip install --upgrade pip
if errorlevel 1 goto fail
venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 goto fail
venv\Scripts\python -m pip install --no-deps rapidocr==3.10.0
if errorlevel 1 goto fail
echo ok> venv\ready.txt
copy /y requirements.txt venv\req.txt >nul

:run
echo [3/3] Starting ... browser will open.
start "" http://127.0.0.1:7860
venv\Scripts\python app.py
if errorlevel 1 goto fail
pause
exit /b 0

:try
if defined PY exit /b 0
%* -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
if not errorlevel 1 set PY=%*
exit /b 0

:fail
echo.
echo ========================================
echo  ERROR - capture this window and send it
echo ========================================
pause
exit /b 1
