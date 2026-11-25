@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Image Translator
if not exist python\python.exe (
  echo Run the installer first.
  pause
  exit /b 1
)
set PYTHONUTF8=1
set PIP_NO_CACHE_DIR=1
set TEMP=%~dp0tmp
set TMP=%~dp0tmp
if not exist tmp mkdir tmp
echo Checking for updates ...
python\python.exe tools\update.py update
echo Starting ... browser will open.
start "" http://127.0.0.1:7870
python\python.exe web.py
if errorlevel 1 pause
exit /b 0
