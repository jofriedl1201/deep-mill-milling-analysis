import os
import subprocess
import sys

NVCC_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe"
ENV_VAR_NAME = "CUDA_PATH_V12_1"

def verify():
    print("--- CUDA 12.1 Verification ---")
    
    # 1. Check Existence
    if os.path.exists(NVCC_PATH):
        print(f"[OK] Found nvcc.exe at: {NVCC_PATH}")
    else:
        print(f"[FAIL] nvcc.exe NOT found at: {NVCC_PATH}")
        return False

    # 2. Invoke Explicitly
    try:
        print(f"Invoking: \"{NVCC_PATH}\" --version")
        result = subprocess.run([NVCC_PATH, "--version"], capture_output=True, text=True)
        print("--- Output ---")
        print(result.stdout)
        if "release 12.1" in result.stdout:
            print("[OK] Confirmed version 12.1")
        else:
            print("[WARN] Version string mismatch (Check output)")
    except Exception as e:
        print(f"[FAIL] execution failed: {e}")
        return False
        
    # 3. Check Environment Variable
    val = os.environ.get(ENV_VAR_NAME)
    if val:
        print(f"[OK] Environment variable {ENV_VAR_NAME} found: {val}")
    else:
        print(f"[INFO] Environment variable {ENV_VAR_NAME} NOT set in current session (Expected if no restart).")

    return True

if __name__ == "__main__":
    verify()
