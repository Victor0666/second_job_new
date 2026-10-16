@echo off
cd /d "%~dp0"
set "PY=D:\Python3.12\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m venv --system-site-packages .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -c "import torch,numpy,networkx,gym,pandas,matplotlib,scipy,openpyxl"
if errorlevel 1 .venv\Scripts\python.exe -m pip install -r requirements.txt
