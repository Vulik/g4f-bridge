@echo off
REM g4f-Bridge Setup Script for Windows
REM Run: setup.bat

echo.
echo ================================================================
echo               g4f-Bridge Setup for Windows
echo ================================================================
echo.

REM Check for Python
echo Checking Python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

python --version

REM Create project directory
echo.
echo Setting up project directory...
set PROJECT_DIR=%USERPROFILE%\g4f-bridge
mkdir "%PROJECT_DIR%" 2>nul
cd /d "%PROJECT_DIR%"

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
pip install --upgrade pip setuptools wheel

REM Install dependencies
echo.
echo Installing dependencies...
pip install fastapi uvicorn[standard] g4f aiohttp requests aiosqlite apscheduler flask

REM Try optional packages
echo.
echo Installing optional dependencies...
pip install tiktoken 2>nul && echo   tiktoken installed || echo   tiktoken failed (using fallback)
pip install orjson 2>nul && echo   orjson installed || echo   orjson failed

REM Create startup script
echo.
echo Creating startup script...
(
echo @echo off
echo cd /d "%%~dp0"
echo call venv\Scripts\activate.bat
echo python main.py
echo pause
) > "%PROJECT_DIR%\start.bat"

REM Print completion message
echo.
echo ================================================================
echo                    Setup Complete!
echo ================================================================
echo.
echo To start the bridge:
echo.
echo   cd %PROJECT_DIR%
echo   start.bat
echo.
echo Or double-click start.bat in the project folder.
echo.

REM Offer to start now
set /p START="Start g4f-Bridge now? (y/n) "
if /i "%START%"=="y" (
    cd /d "%PROJECT_DIR%"
    call venv\Scripts\activate.bat
    python main.py
)

pause