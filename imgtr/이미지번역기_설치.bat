@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Image Translator - Install
echo ==================================================
echo   Image Translator - Install
echo   Everything goes into this folder only:
echo   %~dp0
echo ==================================================
set PYTHONUTF8=1
set PIP_NO_CACHE_DIR=1
set TEMP=%~dp0tmp
set TMP=%~dp0tmp
if not exist tmp mkdir tmp

rem ---- Microsoft Visual C++ runtime (needed by the graphics-card packages) ----
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
if not errorlevel 1 goto vcok
echo.
echo [0/2] Installing Microsoft Visual C++ runtime ... (click YES if Windows asks)
curl -L --fail -o tmp\vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe
if errorlevel 1 goto fail
tmp\vc_redist.x64.exe /install /quiet /norestart
del tmp\vc_redist.x64.exe
:vcok

if exist python\python.exe goto app

echo.
echo [1/2] Downloading Python (portable, into this folder) ...
curl -L --fail -o tmp\py.zip https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip
if errorlevel 1 goto fail
if not exist python mkdir python
tar -xf tmp\py.zip -C python
if errorlevel 1 goto fail
del tmp\py.zip
> python\python312._pth echo python312.zip
>> python\python312._pth echo .
>> python\python312._pth echo ..
>> python\python312._pth echo import site
curl -L --fail -o tmp\get-pip.py https://bootstrap.pypa.io/get-pip.py
if errorlevel 1 goto fail
python\python.exe tmp\get-pip.py --no-warn-script-location
if errorlevel 1 goto fail

:app
echo.
echo [2/2] Downloading the program and packages ...
if not exist tools mkdir tools
curl -L --fail -o tools\update.py https://raw.githubusercontent.com/papa6229-beep/mosaic/main/imgtr/tools/update.py
if errorlevel 1 goto fail
python\python.exe tools\update.py install
if errorlevel 1 goto fail
echo.
pause
exit /b 0

:fail
echo.
echo ========================================
echo  ERROR - capture this window and send it
echo ========================================
pause
exit /b 1
