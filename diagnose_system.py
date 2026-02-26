import subprocess
import sys
import os
import platform

def run_cmd(cmd, check=False):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def main():
    print("=== System Diagnosis Report ===")
    
    # 1. GPU & Driver
    print("\n[1] GPU & Driver:")
    smi = run_cmd("nvidia-smi")
    if smi:
        print("NVIDIA-SMI detected. Output snippet:")
        for line in smi.split('\n')[:10]: # Print header info
            if "Version" in line or "GeForce" in line or "RTX" in line:
                print(f"  {line.strip()}")
    else:
        print("MISSING: nvidia-smi not found or failed.")

    # 2. CUDA Toolkit
    print("\n[2] CUDA Toolkit:")
    nvcc = run_cmd("nvcc --version")
    if nvcc:
        print(f"PRESENT: {nvcc.splitlines()[-1]}")
    else:
        print("MISSING: nvcc (CUDA Toolkit) not found in PATH.")

    # 3. cuDNN
    print("\n[3] cuDNN:")
    # Check common environment variables or headers? 
    # Hard to detect without checking files, but we can look at path
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        print(f"CUDA_PATH detected: {cuda_path}")
        # Simplistic check for cudnn.h or dll
        bin_path = os.path.join(cuda_path, "bin")
        if os.path.exists(bin_path):
            dlls = [f for f in os.listdir(bin_path) if "cudnn" in f]
            if dlls:
                print(f"PRESENT: Found cuDNN DLLs: {', '.join(dlls[:3])}...")
            else:
                print("MISSING: No cuDNN DLLs found in CUDA_PATH/bin.")
        else:
            print("WARNING: CUDA_PATH/bin does not exist.")
    else:
        print("MISSING: CUDA_PATH environment variable not set.")

    # 4. PyTorch
    print("\n[4] PyTorch:")
    try:
        import torch
        print(f"Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("MISSING: PyTorch not installed.")
    except Exception as e:
        print(f"ERROR checking PyTorch: {e}")

    # 5. Visual Studio Build Tools
    print("\n[5] Visual Studio Build Tools:")
    # Try vswhere
    vswhere = "c:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe"
    if os.path.exists(vswhere):
        vs_out = run_cmd(f'"{vswhere}" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath')
        if vs_out:
            print(f"PRESENT: VS/BuildTools detected at: {vs_out}")
        else:
            print("MISSING: vswhere found but no VC Tools detected.")
    else:
        print("MISSING: vswhere.exe not found (VS Installer likely not present).")

    # 6. CMake
    print("\n[6] CMake:")
    cmake = run_cmd("cmake --version")
    if cmake:
        print(f"PRESENT: {cmake.splitlines()[0]}")
    else:
        print("MISSING: cmake not found in PATH.")

    print("\n===============================")

if __name__ == "__main__":
    main()
