@echo off
REM ============================================================
REM  ESG Report Intelligence - start the app on Windows
REM  Double-click this file. The window stays open so you can
REM  see progress and any error message.
REM ============================================================

REM Re-open this script inside "cmd /k" so the window never closes by itself.
if /i "%~1"=="--inner" goto :main
cmd /k ""%~f0" --inner"
exit /b

:main
cd /d "%~dp0"
title ESG Report Intelligence

if not exist "requirements.txt" goto :not_extracted
if not exist "app.py" goto :not_extracted

REM ---------- find Python ----------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto :no_python

REM ---------- virtual environment ----------
if exist ".venv\Scripts\python.exe" goto :have_venv
echo [1/4] Creating the virtual environment ...
%PY% -m venv .venv
if errorlevel 1 goto :venv_failed
:have_venv
set "VPY=.venv\Scripts\python.exe"

REM ---------- packages ----------
echo [2/4] Installing packages - the first run takes a few minutes ...
"%VPY%" -m pip install --upgrade pip --disable-pip-version-check -q
"%VPY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 goto :pip_failed

REM ---------- API key file ----------
if exist ".env" goto :have_env
copy ".env.example" ".env" >nul
echo.
echo [3/4] A .env file was created and opens in Notepad now.
echo       Paste your Gemini key after GEMINI_API_KEY= then SAVE and CLOSE Notepad.
echo       Free key: https://aistudio.google.com/apikey
notepad ".env"
:have_env

REM ---------- skip Streamlit's first-run e-mail question ----------
if exist "%USERPROFILE%\.streamlit\credentials.toml" goto :creds_ok
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
> "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
>> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
:creds_ok

REM ---------- start ----------
echo.
echo [4/4] Starting the app - your browser opens at http://localhost:8501
echo       Keep this window open while you use the app. Close it to stop the app.
echo.
"%VPY%" -m streamlit run app.py
echo.
echo The app has stopped. You can close this window.
goto :eof

REM ---------- error messages ----------
:not_extracted
echo.
echo ERROR: app.py / requirements.txt were not found next to this file.
echo Extract the zip first: right-click the zip, choose Extract All,
echo then double-click run_app.bat inside the extracted folder.
goto :eof

:no_python
echo.
echo ERROR: Python was not found.
echo Install Python 3.11 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during installation. Then run this file again.
goto :eof

:venv_failed
echo.
echo ERROR: the virtual environment could not be created.
echo Make sure Python 3.11+ from python.org is installed - not only the Microsoft Store shortcut.
echo Delete the .venv folder in this project if it exists, then run this file again.
goto :eof

:pip_failed
echo.
echo ERROR: package installation failed - see the messages above.
echo Check your internet connection, then run this file again.
goto :eof
