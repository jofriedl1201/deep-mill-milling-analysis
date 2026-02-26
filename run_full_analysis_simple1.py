import cadquery as cq
import sys
import os

# Fix encoding for Unicode characters
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

step_file = 'Step Files/simple 1.step'
mesh_file = 'temp_simple1.stl'

print("=" * 60)
print("STEP 1: Generating Mesh from STEP")
print("=" * 60)

if not os.path.exists(step_file):
    print(f"Error: {step_file} not found.")
    sys.exit(1)

print(f"Loading {step_file}...")
try:
    obj = cq.importers.importStep(step_file)
    print("Exporting to STL...")
    cq.exporters.export(obj, mesh_file)
    print(f"SUCCESS: Created {mesh_file}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 2: Running Full Analysis Engine")
print("=" * 60)

# Now run the analysis engine
from analysis_engine.main import main as analysis_main

# Override sys.argv for the analysis_main function
original_argv = sys.argv.copy()
sys.argv = ['analysis_main', step_file, mesh_file]

try:
    analysis_main()
except SystemExit as e:
    # analysis_main calls sys.exit(), catch it to continue
    if e.code != 0:
        print(f"\nAnalysis exited with code: {e.code}")
finally:
    sys.argv = original_argv

print("\n" + "=" * 60)
print("COMPLETE!")
print("=" * 60)
