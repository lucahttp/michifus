"""
Download Gemma 4 E2B QAT GGUF using Windows Python (no bash/WSL).
Run: python download_hf.py
"""
import os
import sys
import ctypes
import subprocess

# Force CREATE_NO_WINDOW to avoid shell issues
CREATE_NO_WINDOW = 0x08000000

def run(args, **kwargs):
    kwargs.setdefault('creationflags', CREATE_NO_WINDOW)
    kwargs.setdefault('shell', False)
    kwargs.setdefault('capture_output', True)
    return subprocess.run(args, **kwargs)

print("[1/3] Installing dependencies...")

# Check Python
result = run([sys.executable, "--version"])
print(f"Python: {result.stdout.decode().strip()}")

# Install huggingface_hub
result = run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hup"])
if result.returncode != 0:
    print(f"pip install failed: {result.stderr.decode()}")
else:
    print("huggingface_hub installed")

# Actually install it first
result = run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
print(f"pip: {result.returncode}")

print("[2/3] Downloading Gemma 4 E2B QAT GGUF...")

HF_TOKEN = "YOUR_HF_TOKEN"
LOCAL_DIR = r"E:\huggingface\hub\models--unsloth--gemma-4-E2B-it-qat-mobile-GGUF"

os.makedirs(LOCAL_DIR, exist_ok=True)

# Use huggingface_hub to download
download_script = f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="unsloth/gemma-4-E2B-it-qat-mobile-GGUF",
    local_dir=r"{LOCAL_DIR}",
    token="{HF_TOKEN}",
    local_dir_use_symlinks=False
)
print("DOWNLOAD_COMPLETE")
"""

result = run([sys.executable, "-c", download_script])
print(result.stdout.decode())
if result.stderr:
    print("STDERR:", result.stderr.decode()[:500])

print("[3/3] Verifying...")

# List files
for f in os.listdir(LOCAL_DIR):
    fp = os.path.join(LOCAL_DIR, f)
    if os.path.isfile(fp):
        size = os.path.getsize(fp)
        print(f"  {f} — {size/1024/1024:.1f} MB")

print("Done!")
