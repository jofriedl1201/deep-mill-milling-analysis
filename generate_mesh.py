import cadquery as cq
import sys
import os

step_path = r"Step Files/Left side Traitener.step"
out_path = "validation_mesh.stl"

if not os.path.exists(step_path):
    print(f"Error: {step_path} not found.")
    sys.exit(1)

print(f"Loading {step_path}...")
try:
    obj = cq.importers.importStep(step_path)
    print("Exporting STL...")
    cq.exporters.export(obj, out_path)
    print(f"Created {out_path}")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
