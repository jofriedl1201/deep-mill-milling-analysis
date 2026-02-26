import torch
print(f"Torch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
try:
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
except:
    print("CUDA Device: None")
