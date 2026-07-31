import sys
print("=== Python ===", flush=True)
print(sys.version)

print("\n=== Torch ===", flush=True)
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)}")
        print(f"capability={torch.cuda.get_device_capability(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"vram_free={free/1e9:.1f}GB total={total/1e9:.1f}GB")
except ImportError as e:
    print(f"NO TORCH: {e}")

print("\n=== Other deps ===", flush=True)
for pkg in ["transformers", "datasets", "accelerate", "huggingface_hub"]:
    try:
        m = __import__(pkg)
        print(f"{pkg}={m.__version__}")
    except ImportError:
        print(f"{pkg}=MISSING")
