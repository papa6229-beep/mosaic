@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set CUT_MIN=5
set CUT_AUDIO=1
set CUT_ASK=1
set PYEXE=venv\Scripts\python.exe
if exist python\python.exe set PYEXE=python\python.exe
if not exist %PYEXE% (
  echo Run start.bat first.
  pause
  exit /b 1
)
%PYEXE% tools\cut.py %*
