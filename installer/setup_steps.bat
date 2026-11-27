@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=mbcs:replace
set PIP_NO_CACHE_DIR=1
set PIP_PROGRESS_BAR=off
set TEMP=%~dp0tmp
set TMP=%~dp0tmp
if not exist tmp mkdir tmp
goto %1

:vc
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
if not errorlevel 1 (echo Visual C++ runtime: already installed & exit /b 0)
echo Downloading Microsoft Visual C++ runtime ...
curl -sS -L --fail -o tmp\vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe || exit /b 1
tmp\vc_redist.x64.exe /install /quiet /norestart
del tmp\vc_redist.x64.exe
exit /b 0

:python
if exist python\python.exe (echo Python: already here & exit /b 0)
echo Downloading Python 3.12 (portable, into this folder) ...
curl -sS -L --fail -o tmp\py.zip https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip || exit /b 1
if not exist python mkdir python
tar -xf tmp\py.zip -C python || exit /b 1
del tmp\py.zip
> python\python312._pth echo python312.zip
>> python\python312._pth echo .
>> python\python312._pth echo ..
>> python\python312._pth echo import site
curl -sS -L --fail -o tmp\get-pip.py https://bootstrap.pypa.io/get-pip.py || exit /b 1
python\python.exe tmp\get-pip.py --no-warn-script-location || exit /b 1
exit /b 0

:fetch
if not exist tools mkdir tools
curl -sS -L --fail -o tools\update.py https://raw.githubusercontent.com/papa6229-beep/mosaic/main/tools/update.py || exit /b 1
python\python.exe -u tools\update.py fetch || exit /b 1
exit /b 0

:packages
python\python.exe -u tools\update.py packages || exit /b 1
exit /b 0

:finish
python\python.exe -u tools\update.py finish %2 || exit /b 1
rem let normal users write here (jobs, settings, updates)
icacls "%~dp0." /grant *S-1-5-32-545:(OI)(CI)M /T /C /Q >nul 2>&1
exit /b 0
