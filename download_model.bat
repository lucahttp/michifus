@echo off
echo Downloading Gemma 4 E2B QAT GGUF...
"D:\michi-adapter\venv\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/gemma-4-E2B-it-qat-mobile-GGUF', token='YOUR_HF_TOKEN')"
echo Done!
pause
