@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Mosaic Generator
if exist python\python.exe goto portable
set REPO=https://github.com/papa6229-beep/mosaic

rem ---- Git: auto install if missing ----
set GIT=
where git >nul 2>&1 && set GIT=git
if not defined GIT if exist "%ProgramFiles%\Git\cmd\git.exe" set GIT="%ProgramFiles%\Git\cmd\git.exe"
if defined GIT goto update
echo Installing Git ...
winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
set GIT="%ProgramFiles%\Git\cmd\git.exe"

rem ---- Get latest version ----
:update
echo Checking for updates ...
if not exist .git %GIT% init -q
if not exist .git goto run
%GIT% remote remove origin >nul 2>&1
%GIT% remote add origin %REPO%
%GIT% fetch -q --depth 1 origin main
if errorlevel 1 goto noupdate
%GIT% reset -q --hard origin/main
echo Up to date.
goto run
:noupdate
echo Update check failed - running current version.

:run
call run.bat
exit /b 0

rem ---- Installed version: portable Python in this folder, updates without Git ----
:portable
set PYTHONUTF8=1
set PIP_NO_CACHE_DIR=1
set TEMP=%~dp0tmp
set TMP=%~dp0tmp
if not exist tmp mkdir tmp
echo Checking for updates ...
python\python.exe tools\update.py update
echo Starting ... browser will open.
start "" http://127.0.0.1:7860
python\python.exe app.py
if errorlevel 1 pause
exit /b 0
