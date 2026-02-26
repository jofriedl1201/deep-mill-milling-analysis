import zipfile
import os
import shutil
import sys

ZIP_PATH = r"c:\Users\jfrie\dev\Milling Analysis\cudnn_8.9.7_cuda12.zip"
DEST_ROOT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
EXTRACT_DIR = r"c:\Users\jfrie\dev\Milling Analysis\cudnn_extract_temp"

def install():
    print(f"Extracting {ZIP_PATH}...")
    try:
        if os.path.exists(EXTRACT_DIR):
            shutil.rmtree(EXTRACT_DIR)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

    # Find the root of the extracted content
    source_root = None
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if "bin" in dirs and "include" in dirs:
            # check for cudnn file to be sure
            if os.path.exists(os.path.join(root, "bin")):
                source_root = root
                break
    
    if not source_root:
        print("Could not find valid cuDNN structure in extracted files.")
        return False

    print(f"Found source root: {source_root}")

    # Define mappings
    # (relative_src, destination_abs)
    mappings = [
        (os.path.join("bin"), os.path.join(DEST_ROOT, "bin")),
        (os.path.join("include"), os.path.join(DEST_ROOT, "include")),
        (os.path.join("lib", "x64"), os.path.join(DEST_ROOT, "lib", "x64"))
    ]

    failed_files = []

    for rel_src, abs_dest in mappings:
        src_dir = os.path.join(source_root, rel_src)
        if not os.path.exists(src_dir):
            print(f"Warning: Source dir {src_dir} does not exist.")
            continue
            
        if not os.path.exists(abs_dest):
            # Try to create if missing (unlikely for CUDA dir)
            try:
                os.makedirs(abs_dest)
            except OSError:
                print(f"Could not create dest dir: {abs_dest}")
                failed_files.append(f"DIR: {abs_dest}")
                continue

        print(f"Copying from {rel_src} to {abs_dest}...")
        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(abs_dest, filename)
            
            if os.path.isfile(src_file):
                try:
                    shutil.copy2(src_file, dest_file)
                    print(f"  [OK] {filename}")
                except PermissionError:
                    print(f"  [FAIL] Permission denied: {filename}")
                    failed_files.append(dest_file)
                except Exception as e:
                    print(f"  [FAIL] Error copying {filename}: {e}")
                    failed_files.append(dest_file)

    if failed_files:
        print("\n--- INSTALLATION FAILED (PARTIAL) ---")
        print("Could not copy the following files (likely Permission Denied):")
        for f in failed_files:
            print(f" - {f}")
        print("\nPlease copy these files manually.")
        return False
    
    # Verify
    print("\n--- VERIFICATION ---")
    
    # Check key files
    check_files = [
        os.path.join(DEST_ROOT, "bin", "cudnn64_8.dll"),
        os.path.join(DEST_ROOT, "include", "cudnn.h"),
        os.path.join(DEST_ROOT, "lib", "x64", "cudnn.lib")
    ]
    
    all_ok = True
    for f in check_files:
        if os.path.exists(f):
             print(f"[OK] Found {os.path.basename(f)}")
        else:
             print(f"[MISSING] {os.path.basename(f)}")
             all_ok = False
             
    if all_ok:
        print("SUCCESS: cuDNN 8.9 Installed and Verified.")
        return True
    else:
        print("Verification failed: Some files are missing.")
        return False

if __name__ == "__main__":
    if install():
        sys.exit(0)
    else:
        sys.exit(1)
