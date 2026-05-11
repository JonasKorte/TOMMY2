@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python is not installed or not on PATH.
  echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/
  echo and tick "Add Python to PATH" during install.
  pause
  exit /b 1
)

if not exist venv (
  echo [setup] Creating virtual environment...
  python -m venv venv
)
call venv\Scripts\activate.bat

echo [setup] Installing/updating Python dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
  echo [ERROR] pip install failed. Check your internet connection.
  pause
  exit /b 1
)

if not exist "models\llm\Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" (
  echo [setup] First-run model download (~6 GB total). This happens once.
  python scripts\download_models.py
  if errorlevel 1 (
    echo [ERROR] Model download failed.
    pause
    exit /b 1
  )
)

python run.py
pause
