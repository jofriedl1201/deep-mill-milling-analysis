import os
import sys

SEARCH_ROOT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

def find_nvcc():
    print(f"Searching for nvcc.exe in {SEARCH_ROOT}...")
    try:
        if not os.path.exists(SEARCH_ROOT):
            print("CUDA Root not found.")
            return

        for root, dirs, files in os.walk(SEARCH_ROOT):
            if "nvcc.exe" in files:
                full_path = os.path.join(root, "nvcc.exe")
                print(f"FOUND: {full_path}")
                # Print version if possible
                try:
                    import subprocess
                    out = subprocess.check_output([full_path, "--version"], text=True)
                    print(out)
                except:
                    pass
                return
        print("nvcc.exe NOT FOUND in CUDA root.")
    except Exception as e:
        print(f"Error searching: {e}")

if __name__ == "__main__":
    find_nvcc()
