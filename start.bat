@echo off
cd /d %~dp0

REM Prefer venv python if exists
if exist .venv\Scripts\python.exe (
  set PY=.venv\Scripts\python.exe
) else (
  set PY=python
)

"%PY%" launcher.py

pause


