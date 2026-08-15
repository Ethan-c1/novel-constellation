@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

:: Novel Constellation launcher for Windows

cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "BOOTSTRAP_PYTHON="

if not exist "%VENV_PYTHON%" (
    echo [1/3] Creating a local Python environment...

    where py >nul 2>nul
    if not errorlevel 1 set "BOOTSTRAP_PYTHON=py -3"

    if not defined BOOTSTRAP_PYTHON (
        where python >nul 2>nul
        if not errorlevel 1 set "BOOTSTRAP_PYTHON=python"
    )

    if not defined BOOTSTRAP_PYTHON goto :python_missing

    !BOOTSTRAP_PYTHON! -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if errorlevel 1 goto :python_too_old

    !BOOTSTRAP_PYTHON! -m venv "%VENV_DIR%"
    if errorlevel 1 goto :setup_failed
)

echo [2/3] Installing or checking dependencies...
"%VENV_PYTHON%" -m pip install --disable-pip-version-check -r "backend\requirements.txt"
if errorlevel 1 goto :setup_failed

if not exist ".env" (
    copy /Y ".env.example" ".env" >nul
    echo.
    echo A local .env file has been created.
    echo Please fill in LLM_API_KEY, save the file, and run start.bat again.
    start "" notepad.exe ".env"
    pause
    exit /b 0
)

echo [3/3] Starting Novel Constellation...
"%VENV_PYTHON%" -u "start.py"
exit /b %errorlevel%

:python_missing
echo.
echo [ERROR] Python 3.10 or newer was not found.
echo Install Python from https://www.python.org/downloads/ and enable "Add Python to PATH".
pause
exit /b 1

:python_too_old
echo.
echo [ERROR] Python 3.10 or newer is required.
pause
exit /b 1

:setup_failed
echo.
echo [ERROR] Failed to create the environment or install dependencies.
echo Check your network connection and try again.
pause
exit /b 1
