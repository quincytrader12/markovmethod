@echo off
REM ===========================================================================
REM  Mamba Terminal - one-click Windows build
REM
REM  Double-click this file (build.bat) on a Windows PC to build the app into
REM  standalone .exe files. It does exactly what the cloud build did:
REM      1. installs Mamba Terminal + build tools into a local environment
REM      2. builds mamba-web.exe   (the neon web HUD)
REM      3. builds mamba-terminal.exe (the text terminal)
REM
REM  You only need Python 3.12 installed first (see BUILD_WINDOWS.md).
REM  The finished .exe files land in the "dist" folder next to this file.
REM ===========================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   Mamba Terminal - Windows build
echo ============================================================
echo.

REM --- Find Python --------------------------------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python was not found on this PC.
  echo.
  echo   1^) Install Python 3.12 from https://www.python.org/downloads/
  echo   2^) During install, TICK the box "Add Python to PATH"
  echo   3^) Then double-click this file again.
  echo.
  pause
  exit /b 1
)

echo Using Python:
%PY% --version
echo.

REM --- Create a clean local build environment -----------------------------
if not exist ".buildenv" (
  echo Creating build environment ^(one-time^)...
  %PY% -m venv .buildenv
  if errorlevel 1 ( echo [ERROR] Could not create the build environment. & pause & exit /b 1 )
)
call ".buildenv\Scripts\activate.bat"

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing Mamba Terminal + build tools ^(this can take a few minutes^)...
pip install ".[terminal,web,build]"
if errorlevel 1 ( echo. & echo [ERROR] Install step failed. See the messages above. & pause & exit /b 1 )

echo.
echo Building mamba-terminal.exe ...
pyinstaller --clean --noconfirm packaging\mamba-terminal.spec
if errorlevel 1 ( echo. & echo [ERROR] terminal build failed. See the messages above. & pause & exit /b 1 )

echo.
echo Building mamba-web.exe ^(neon web HUD^) ...
pyinstaller --clean --noconfirm packaging\mamba-web.spec
if errorlevel 1 ( echo. & echo [ERROR] web build failed. See the messages above. & pause & exit /b 1 )

echo.
echo ============================================================
echo   DONE.  Your apps are in the "dist" folder:
echo.
echo     dist\mamba-web.exe       ^<-- neon web HUD, double-click to run
echo     dist\mamba-terminal.exe  ^<-- text terminal
echo ============================================================
echo.
pause
