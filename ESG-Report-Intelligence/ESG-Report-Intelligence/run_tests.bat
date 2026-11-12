@echo off
REM Runs the automated test suite (no API key needed - uses the offline test provider)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m pytest -q
pause
