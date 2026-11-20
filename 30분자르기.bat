@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set CUT_MIN=30
if not exist venv\Scripts\python.exe (
  echo Run start.bat first.
  pause
  exit /b 1
)
venv\Scripts\python tools\cut.py %*
