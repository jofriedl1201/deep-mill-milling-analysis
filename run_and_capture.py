import subprocess
import os

env = os.environ.copy()
env["PYTHONPATH"] = r"c:\Users\jfrie\dev\Milling Analysis"

cmd = [
    "python", 
    "analysis_engine/main.py", 
    r"c:\Users\jfrie\dev\Milling Analysis\Step Files\Left side Traitener.step",
    r"c:\Users\jfrie\dev\cad_intelligence_engine\public\models\default_part.stl"
]

with open("run_output.txt", "w") as f:
    subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
