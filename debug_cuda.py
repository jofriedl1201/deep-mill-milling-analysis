import os
import subprocess

path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
print(f"Listing {path}...")
try:
    if os.path.exists(path):
        for item in os.listdir(path):
            print(f"FOUND: {item}")
    else:
        print("CUDA Directory not found.")
        
    # Check env
    print(f"ENV CUDA_PATH: {os.environ.get('CUDA_PATH')}")
    
    # Try where
    try:
        out = subprocess.check_output(["where", "nvcc"], text=True)
        print(f"WHERE NVCC: {out}")
    except:
        print("where nvcc failed")
        
except Exception as e:
    print(e)
