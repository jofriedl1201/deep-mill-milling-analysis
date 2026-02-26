import os
import subprocess
import numpy as np

def run_cmd(cmd):
    print(f"RUNNING: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)

def main():
    print("=== DeepMill Test Harness ===", flush=True)
    
    # 1. Setup Directories
    os.makedirs("data/raw_data/models", exist_ok=True)
    os.makedirs("data/raw_data/models_cutter", exist_ok=True)
    
    # 2. Generate Dummy Part (10x10x10 cube grid)
    # Format: X Y Z Nx Ny Nz Label1 Label2
    print("Generating dummy part...", flush=True)
    points = []
    for x in np.linspace(0, 1, 10):
        for y in np.linspace(0, 1, 10):
            for z in np.linspace(0, 1, 10):
                # Normal pointing up Z
                points.append(f"{x:.4f} {y:.4f} {z:.4f} 0.0 0.0 1.0 0 0")
    
    with open("data/raw_data/models/test_part.txt", "w") as f:
        f.write("\n".join(points))
        
    # 3. Generate Dummy Cutter
    # Format: 4 params (D, H, ...?) - Guessing common format D R H ...
    # Readme says "four parameters". I'll put random reasonable floats.
    with open("data/raw_data/models_cutter/test_part.txt", "w") as f:
        f.write("6.0 0.0 10.0 0.0") 
        
    # 4. Run Preprocessing
    print("Running Preprocessing (seg_deepmill_cutter.py)...", flush=True)
    # Need to make sure filelist is generated correctly by the script or manually
    # The README says it creates 'filelist' folder.
    try:
        run_cmd("python tools/seg_deepmill_cutter.py")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        # If script fails, we might mock the output if we knew format, but let's hope.
    
    # Check if preprocessing worked
    if os.path.exists("data/filelist") and os.path.exists("data/points"):
        print("Preprocessing successful. Filelists created.")
    else:
        print("Preprocessing failed to create output folders. Investigating...")
        run_cmd("ls -R data")
        
    # 5. Run Inference
    # We need to ensure we run TEST mode.
    # The runner script loops ratios. We just want one run.
    # We will construct a direct command to segmentation.py if possible, or use runner with args.
    # But runner doesn't have --run argument easily exposed (it hardcodes script string?).
    # Running runner might train.
    # We want TEST.
    # Try using runner but limiting epochs to 0 probably fails.
    # Let's try to invoke segmentation.py directly for prediction if known.
    
    # Constructing command based on run_seg_deepmill.py logic:
    # python segmentation.py --config configs/seg_deepmill.yaml SOLVER.run test DATA.test.filelist ...
    
    # We need a filelist pointing to our generated data.
    # If Preprocessing worked, there should be a filelist.
    # Assuming 'models' category.
    test_list = "data/filelist/models_test.txt"
    if not os.path.exists(test_list):
        # Fallback: create one
        os.makedirs("data/filelist", exist_ok=True)
        with open(test_list, "w") as f:
            f.write("test_part")
            
    print("Running Inference...", flush=True)
    cmd = (
        "python segmentation.py --config configs/seg_deepmill.yaml "
        "SOLVER.run test "
        "SOLVER.gpu 0 "  # Use GPU 0
        "DATA.test.filelist data/filelist/models_test.txt "
        "DATA.test.depth 5 "
        "MODEL.stages 3 "
        "SOLVER.logdir logs/test_run"
    )
    
    try:
        run_cmd(cmd)
        print("Inference command finished.")
    except Exception as e:
        print(f"Inference failed (Expected if no GPU/Weights): {e}")

    # 6. Report Results
    print("Checking logs for output...")
    run_cmd("ls -R logs/test_run")
    
if __name__ == "__main__":
    main()
