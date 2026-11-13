import torch

print("Python aur PyTorch check kar rahe hain...")
print("PyTorch Version:", torch.__version__)

if torch.cuda.is_available():
    print("✅ GPU mil gaya hai! Naam:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU nahi mila. Sirf CPU par chalega.")