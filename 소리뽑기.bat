@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set CUT_MIN=5
set CUT_AUDIO=1
set CUT_ASK=1
if not exist venv\Scripts\python.exe (
  echo Run start.bat first.
  pause
  exit /b 1
)
venv\Scripts\python tools\cut.py %*
