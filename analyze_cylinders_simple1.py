import sys
sys.path.insert(0, 'BREP Graph Generator')

from brep_step1 import analyze_step_faces

step_file = 'Step Files/simple 1.step'
print(f"Analyzing cylindrical faces in: {step_file}")
print("=" * 60)

result = analyze_step_faces(step_file)

# Filter for cylindrical faces
cylindrical_faces = {}
for face_id, face_data in result['faces'].items():
    if face_data.get('surface_type') == 1:  # Cylinder
        cylindrical_faces[face_id] = face_data

print(f"\nFound {len(cylindrical_faces)} cylindrical faces:\n")

# Display details
for face_id, face_data in sorted(cylindrical_faces.items()):
    geom_params = face_data.get('geom_params', {})
    
    radius = geom_params.get('radius', 'Unknown')
    origin = geom_params.get('origin', 'Unknown')
    axis = geom_params.get('axis_direction', 'Unknown')
    area = face_data.get('area', 'Unknown')
    
    print(f"Face {face_id}:")
    print(f"  Radius: {radius:.4f} mm" if isinstance(radius, (int, float)) else f"  Radius: {radius}")
    print(f"  Surface Area: {area:.2f} mm²" if isinstance(area, (int, float)) else f"  Surface Area: {area}")
    
    if isinstance(origin, (list, tuple)) and len(origin) >= 3:
        print(f"  Origin: ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")
    else:
        print(f"  Origin: {origin}")
    
    if isinstance(axis, (list, tuple)) and len(axis) >= 3:
        print(f"  Axis: ({axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f})")
    else:
        print(f"  Axis: {axis}")
    
    # Estimate height from area and radius
    # Area = 2πrh (excluding caps)
    if isinstance(radius, (int, float)) and isinstance(area, (int, float)) and radius > 0:
        import math
        estimated_height = area / (2 * math.pi * radius)
        print(f"  Estimated Height: {estimated_height:.2f} mm")
    
    print()

print("=" * 60)
