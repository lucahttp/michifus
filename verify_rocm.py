import torch

print("=" * 60)
print("  OFFICIAL AMD ROCm 7.2.1 WINDOWS GPU VERIFICATION TEST")
print("=" * 60)
print("PyTorch Version:", torch.__version__)
print("ROCm / CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Device Name [0]:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM Memory: Free={free/1e9:.2f} GB / Total={total/1e9:.2f} GB")
else:
    print("Device: CPU Fallback")
print("=" * 60)
