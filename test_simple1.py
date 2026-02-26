import sys
sys.path.insert(0, 'BREP Graph Generator')

from brep_step1 import analyze_step_faces

step_file = 'Step Files/simple 1.step'
print(f"Analyzing: {step_file}")
print("=" * 60)

result = analyze_step_faces(step_file)

print(f"\nAnalysis Results:")
print(f"  Faces: {len(result['faces'])}")
print(f"  Edges: {len(result['edges'])}")

# Show face types
face_types = {}
for face_id, face_data in result['faces'].items():
    surface_type = face_data.get('surface_type', 'Unknown')
    face_types[surface_type] = face_types.get(surface_type, 0) + 1

print(f"\nFace Type Breakdown:")
type_names = {0: 'Plane', 1: 'Cylinder', 2: 'Cone', 3: 'Sphere', 4: 'Torus', 5: 'BSpline', 6: 'Bezier', 7: 'Other'}
for st, count in sorted(face_types.items()):
    print(f"  {type_names.get(st, f'Type {st}')}: {count}")

print("\n" + "=" * 60)
print("Graph extraction completed successfully!")
