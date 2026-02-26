"""Test geom_params extraction"""
import sys
sys.path.insert(0, 'BREP Graph Generator')
from brep_step1 import analyze_step_faces_fast

r = analyze_step_faces_fast('Step Files/S-U09900012-000__MOUNT - TADS V2.0 DOOR MOTOR MOUNT.step')
faces = r['faces']

print('=== Dimensional Data from Graph ===')
print()

for i, f in faces.items():
    if f['geom_params']:
        st = f['surface_type']
        gp = f['geom_params']
        st_name = {0:'Plane', 1:'Cylinder', 2:'Cone', 3:'Sphere', 4:'Torus'}.get(st, 'Other')
        
        print(f"Face {i}: {st_name}")
        for k, v in gp.items():
            if isinstance(v, list):
                print(f"  {k}: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]")
            else:
                print(f"  {k}: {v:.3f}")
        print()
