@echo off
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=D:\Python3.12\python.exe"
"%PY%" -X utf8 "%~dp0launch.py" %*
