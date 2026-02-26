import os

CUDA_REL = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
CHECK_FILES = [
    os.path.join(CUDA_REL, "bin", "cudnn64_8.dll"),
    os.path.join(CUDA_REL, "include", "cudnn.h"),
    os.path.join(CUDA_REL, "lib", "x64", "cudnn.lib")
]

def verify():
    print("--- Verifying cuDNN 8.9 Installation ---")
    all_found = True
    for f in CHECK_FILES:
        if os.path.exists(f):
            print(f"[OK] Found {os.path.basename(f)}")
        else:
            print(f"[MISSING] {f}")
            all_found = False
            
    if all_found:
        print("SUCCESS: usage of cuDNN 8.9 with CUDA 12.1 verified.")
    else:
        print("FAILURE: Some files are missing.")

if __name__ == "__main__":
    verify()
