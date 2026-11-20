@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist venv\Scripts\python.exe (
  echo 먼저 start.bat을 한 번 실행해 주세요.
  pause
  exit /b 1
)
venv\Scripts\python tools\cut.py "%~1" 30
