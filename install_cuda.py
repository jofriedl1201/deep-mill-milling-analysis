import urllib.request
import subprocess
import os
import sys

INSTALLER_URL = "https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_windows.exe"
INSTALLER_NAME = "cuda_12.1_installer.exe"

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def install_cuda(filename):
    print("Starting Silent Installation of CUDA 12.1... (This may take a few minutes)")
    # Argument -s for silent. 
    # Install specific components to save time? 
    # Standard: -s
    try:
        # Using -s for silent mode.
        # NOTE: This blocks until finished.
        cmd = [filename, "-s", "compiler_12.1", "runtime_12.1", "cublas_12.1", "cublas_dev_12.1", "cudart_12.1", "curand_12.1", "curand_dev_12.1", "cusolver_12.1", "cusolver_dev_12.1", "cusparse_12.1", "cusparse_dev_12.1", "nvcc_12.1", "visual_studio_integration_12.1"]
        # Actually, standard silent install installs everything by default usually. 
        # Let's try simple "-s" first to avoid missing dependent components for O-CNN.
        subprocess.check_call([filename, "-s"])
        print("Installation command finished.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Installation failed with code {e.returncode}")
        return False

def verify_install():
    print("Verifying nvcc...")
    cuda_path = os.environ.get("CUDA_PATH")
    print(f"CUDA_PATH: {cuda_path}")
    
    try:
        res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        print(res.stdout)
        if "release 12.1" in res.stdout:
            return True
    except FileNotFoundError:
        print("nvcc not found in PATH.")
    
    # Try explicit path if not in global PATH yet (env var update might require shell restart)
    if cuda_path:
        nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")
        if os.path.exists(nvcc_path):
             print(f"Found nvcc at {nvcc_path}")
             return True
             
    return False

if __name__ == "__main__":
    if not os.path.exists(INSTALLER_NAME):
        if not download_file(INSTALLER_URL, INSTALLER_NAME):
            sys.exit(1)
    
    if not install_cuda(INSTALLER_NAME):
        sys.exit(1)
        
    if verify_install():
        print("SUCCESS: CUDA 12.1 Installed and Detected.")
    else:
        print("WARNING: CUDA Installed but nvcc not found in PATH. A restart might be needed.")
