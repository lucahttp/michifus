@echo off
echo ===========================================================
echo   MICHI-ADAPTER: Setup + Run (NO WSL)
echo ===========================================================
echo.

echo [1/4] Checking Python venv...
if not exist "D:\michi-adapter\venv\Scripts\python.exe" (
    echo Creating venv...
    python -m venv D:\michi-adapter\venv
)
echo [OK] Python found

echo.
echo [2/4] Installing core dependencies...
"D:\michi-adapter\venv\Scripts\pip.exe" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
"D:\michi-adapter\venv\Scripts\pip.exe" install --quiet numpy scipy soundfile
echo [OK] Core deps

echo.
echo [3/4] Downloading Gemma 4 E2B QAT GGUF (if needed)...
if not exist "D:\michi-adapter\model_cache" (
    mkdir D:\michi-adapter\model_cache
)
"D:\michi-adapter\venv\Scripts\python.exe" -c "^
from huggingface_hub import snapshot_download;^
snapshot_download('unsloth/gemma-4-E2B-it-qat-mobile-GGUF',^
    local_dir='D:/michi-adapter/model_cache/gemma-4-E2B',^
    token='YOUR_HF_TOKEN')"
echo [OK] Model downloaded

echo.
echo [4/4] Starting Michi-Real live voice server...
echo    Open browser at: http://localhost:8000
echo    Or press Ctrl+C to exit
echo.
"D:\michi-adapter\venv\Scripts\python.exe" "D:\michi-adapter\app_michi_real.py"
pause
