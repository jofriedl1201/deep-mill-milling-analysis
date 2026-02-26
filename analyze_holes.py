"""Proper counterbore/through hole analysis"""
import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE
from OCP.gp import gp_Pnt

# Load part
result = cq.importers.importStep('Step Files/S-U09900012-000__MOUNT - TADS V2.0 DOOR MOTOR MOUNT.step')
solid = result.val()
faces = list(solid.Faces())

# Classify each face with axis info for cylinders
cylinders = []
planes = []

for i, face in enumerate(faces):
    adaptor = BRepAdaptor_Surface(face.wrapped)
    ftype = adaptor.GetType()
    
    if ftype == GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        radius = cyl.Radius()
        # Get cylinder axis location (center point)
        axis_loc = cyl.Location()
        axis_dir = cyl.Axis().Direction()
        cylinders.append({
            'id': i,
            'radius': radius,
            'diameter': 2 * radius,
            'center_x': axis_loc.X(),
            'center_y': axis_loc.Y(),
            'center_z': axis_loc.Z(),
            'axis_dx': axis_dir.X(),
            'axis_dy': axis_dir.Y(),
            'axis_dz': axis_dir.Z()
        })
    elif ftype == GeomAbs_Plane:
        planes.append({'id': i})

print("=== Cylinder Analysis ===")
print(f"Total cylinders: {len(cylinders)}")
print()

# Group cylinders by axis (concentric holes share same axis)
def same_axis(c1, c2, tol=0.1):
    """Check if two cylinders are coaxial (same central axis)"""
    # Check if centers are on same line
    dx = c1['center_x'] - c2['center_x']
    dy = c1['center_y'] - c2['center_y']
    dz = c1['center_z'] - c2['center_z']
    
    # Check axis direction is parallel
    dot = abs(c1['axis_dx']*c2['axis_dx'] + c1['axis_dy']*c2['axis_dy'] + c1['axis_dz']*c2['axis_dz'])
    if dot < 0.99:
        return False
    
    # Check if delta is parallel to axis (coaxial)
    dist = (dx**2 + dy**2 + dz**2)**0.5
    if dist < tol:
        return True
    
    # Project delta onto axis
    proj = abs(dx*c1['axis_dx'] + dy*c1['axis_dy'] + dz*c1['axis_dz'])
    perp_dist = (dist**2 - proj**2)**0.5 if dist > proj else 0
    
    return perp_dist < tol

# Find concentric groups
groups = []
assigned = set()

for i, c1 in enumerate(cylinders):
    if i in assigned:
        continue
    group = [c1]
    assigned.add(i)
    for j, c2 in enumerate(cylinders):
        if j in assigned:
            continue
        if same_axis(c1, c2):
            group.append(c2)
            assigned.add(j)
    groups.append(group)

print("=== Hole Groups (concentric = counterbore) ===")
for gi, group in enumerate(groups):
    diameters = sorted([c['diameter'] for c in group])
    if len(group) > 1:
        print(f"\nGroup {gi+1}: COUNTERBORE (stepped hole)")
        for c in sorted(group, key=lambda x: x['diameter']):
            print(f"  Face {c['id']}: Diameter = {c['diameter']:.3f} mm")
        print(f"  --> Through hole: {min(diameters):.3f} mm, Counterbore: {max(diameters):.3f} mm")
    else:
        c = group[0]
        print(f"\nGroup {gi+1}: SIMPLE HOLE")
        print(f"  Face {c['id']}: Diameter = {c['diameter']:.3f} mm")

print("\n=== Summary ===")
simple_holes = [g for g in groups if len(g) == 1]
counterbores = [g for g in groups if len(g) > 1]

print(f"Simple holes: {len(simple_holes)}")
print(f"Counterbores: {len(counterbores)}")

all_through_diameters = []
for g in groups:
    all_through_diameters.append(min(c['diameter'] for c in g))

print(f"\nSmallest through diameter (drill): {min(all_through_diameters):.3f} mm")
