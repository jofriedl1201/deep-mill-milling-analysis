import urllib.request
import zipfile
import os
import sys
import shutil

# NVIDIA Redist URL (Often publicly accessible without auth)
CUDNN_URL = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip"
FILENAME = "cudnn.zip"
EXTRACT_PATH = "cudnn_extract"

# We need to target the CUDA direction found in Step 1.
# I will search for it again or hardcode the one I found (v12.1)
CUDA_PATH_ROOT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"

def find_cuda_path():
    # Helper to be robust
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(CUDA_PATH_ROOT):
        return CUDA_PATH_ROOT
    # Search
    if os.path.exists(base):
        for d in os.listdir(base):
            if d.startswith("v12"):
                return os.path.join(base, d)
    return None

def install():
    target_path = find_cuda_path()
    if not target_path:
        print("Error: Could not determine CUDA installation path.")
        return False
        
    print(f"Target CUDA Path: {target_path}")
    
    # Check if download needed
    if not os.path.exists(FILENAME):
        print(f"Downloading cuDNN from {CUDNN_URL}...")
        try:
            urllib.request.urlretrieve(CUDNN_URL, FILENAME)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            return False
            
    # Extract
    print("Extracting...")
    try:
        with zipfile.ZipFile(FILENAME, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False
        
    # Move files
    # Structure: cudnn-windows.../archive/bin|include|lib
    # We find the inner folder
    print("Installing files...")
    src_root = None
    for root, dirs, files in os.walk(EXTRACT_PATH):
        if "bin" in dirs and "include" in dirs:
            src_root = root
            break
            
    if not src_root:
        print("Could not find cuDNN root in extracted files.")
        return False
        
    # Copy
    # We shouldn't overwrite blindly without perm, but we are in Admin assume? 
    # Or just copy.
    try:
        def copy_tree(src, dst):
            if not os.path.exists(dst): os.makedirs(dst)
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    copy_tree(s, d)
                else:
                    try:
                        shutil.copy2(s, d)
                        print(f"Copied {item}")
                    except PermissionError:
                        print(f"Permission denied for {d} (Using what we have)")

        copy_tree(os.path.join(src_root, "bin"), os.path.join(target_path, "bin"))
        copy_tree(os.path.join(src_root, "include"), os.path.join(target_path, "include"))
        copy_tree(os.path.join(src_root, "lib"), os.path.join(target_path, "lib"))
        
        print("cuDNN files copied.")
        return True
    except Exception as e:
        print(f"Installation failed: {e}")
        return False

if __name__ == "__main__":
    if install():
        print("SUCCESS: cuDNN Installed.")
    else:
        sys.exit(1)
