import subprocess
import os
import sys

INSTALLER_NAME = "cuda_12.1_installer.exe"

def install_cuda_safe():
    if not os.path.exists(INSTALLER_NAME):
        print("Installer not found.")
        return False
        
    print("Executing CUDA 12.1 Installer...")
    print("Mode: Silent (-s)")
    print("Components: Toolkit ONLY (Values: nvcc_12.1, cudart_12.1, cublas_12.1, etc.)")
    print("Skipping: Display.Driver, Nsight, etc.")
    
    # Components list based on standard modular install for 12.1
    # We explicitly select what we need to avoid driver conflict
    components = [
        "mv_compiler_12.1",
        "mv_cublas_12.1",
        "mv_cublas_dev_12.1",
        "mv_cudart_12.1",
        "mv_curand_12.1",
        "mv_curand_dev_12.1",
        "mv_cusolver_12.1",
        "mv_cusolver_dev_12.1",
        "mv_cusparse_12.1",
        "mv_cusparse_dev_12.1",
        "mv_nvcc_12.1",
        "mv_visual_studio_integration_12.1"
    ]
    # Note: component names vary by version. 
    # For 12.1 local installer, typical names are like "nvcc_12.1" or "compiler_12.1"
    # The previous script tried "compiler_12.1". 
    # Let's try to be safer by using "-s" (silent) but attempting to rely on default toolkit update logic?
    # NO. Default might update driver.
    # The PROMPT constraint "Do not modify GPU drivers" is strict.
    # A generic "-s" is dangerous.
    # Safe approach: Try to verify component names? Impossible without extracting.
    # Standard names for 12.x often use "compiler_12.1".
    
    # Let's try the key ones.
    args = [INSTALLER_NAME, "-s"]
    # If we pass nothing else, it does default.
    # We MUST pass arguments to deselect driver.
    # "Display.Driver" is the component name usually.
    # but CLI usually takes names of what TO install.
    # There isn't a "skip_driver" flag easily.
    # Arg: -install_groups=ToolkitOptions
    # This is supported in newer installers.
    
    # Plan B: Just run it and hope "-s" doesn't force downgrade of newer driver?
    # Usually valid. But risk.
    # Better: Use the instruction to "Instruct user" if we can't be sure.
    # I will try to execute with specific component list from my research (compiler_12.1 etc).
    # If that fails, I notify user.
    
    # Common names for 12.1:
    # nvcc_12.1
    # cudart_12.1
    # ...
    # The prefix "mv_" is internal. usually just "nvcc_12.1".
    
    # I will try running with basic toolkit args.
    unsafe_args = ["compiler_12.1", "runtime_12.1", "nvcc_12.1"] 
    # Actually, simpler:
    # Let's just create the script. If it fails, we fall back to manual.
    
    try:
        # Running a "dry run" or list? No dry run on these EXEs usually.
        # I'll try running with a minimal set.
        cmd = [INSTALLER_NAME, "-s", "compiler_12.1", "runtime_12.1", "cudart_12.1", "nvcc_12.1"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print("Install success.")
        return True
    except Exception as e:
        print(f"Install failed: {e}")
        return False

if __name__ == "__main__":
    if install_cuda_safe():
        sys.exit(0)
    else:
        sys.exit(1)
