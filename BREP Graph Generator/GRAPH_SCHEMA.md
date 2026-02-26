# B-Rep Graph Schema Specification

> [!IMPORTANT]
> This document defines the canonical geometric representation for machining feature reasoning.
> All downstream analysis must consume this schema, not raw Open Cascade geometry.

## Overview

The B-Rep Graph encodes a CAD model's boundary representation as a graph structure where:
- **Nodes** = Faces (topological surfaces)
- **Edges** = Topological adjacencies (shared edges between faces)

---

## Node Features (Per Face)

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `uv_grid` | `(5, 5, 7)` | float32 | UV-sampled point cloud with normals and mask |
| `surface_type` | scalar | int | Surface type ID (see table below) |
| `area` | scalar | float | Surface area in model units² |
| `loop_count` | scalar | int | Number of boundary loops (wires) |
| `adjacency_count` | scalar | int | Number of adjacent faces |

### Surface Type IDs
| ID | Type |
|----|------|
| 0 | Plane |
| 1 | Cylinder |
| 2 | Cone |
| 3 | Sphere |
| 4 | Torus |
| 5 | Bezier Surface |
| 6 | B-Spline Surface |
| 7 | Other |

### UV Grid Structure
Each `uv_grid[u, v]` contains 7 values:
```
[x, y, z, nx, ny, nz, mask]
```
- `x, y, z`: 3D point coordinates
- `nx, ny, nz`: Surface normal at point
- `mask`: 1.0 if point is inside face boundary, 0.0 otherwise

### Geometric Parameters (`geom_params`)

Dimensional data extracted per surface type:

| Surface Type | Parameters |
|--------------|------------|
| **Plane** | `origin [x,y,z]`, `normal [nx,ny,nz]` |
| **Cylinder** | `radius`, `diameter`, `axis_origin [x,y,z]`, `axis_direction [dx,dy,dz]` |
| **Cone** | `half_angle` (radians), `apex [x,y,z]`, `axis_direction [dx,dy,dz]` |
| **Sphere** | `radius`, `center [x,y,z]` |
| **Torus** | `major_radius`, `minor_radius`, `center [x,y,z]`, `axis_direction [dx,dy,dz]` |
| **BSpline/Bezier** | `{}` (empty - no simple params) |

**Example:**
```python
face = graph.faces[0]
if face['surface_type'] == 1:  # Cylinder
    print(f"Diameter: {face['geom_params']['diameter']:.2f} mm")
    print(f"Axis: {face['geom_params']['axis_direction']}")
```


---

## Edge Features (Per Adjacency)

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `curve_grid` | `(5, 6)` | float32 | Sampled points along edge curve |
| `curve_type` | scalar | int | Curve type ID (see table below) |
| `length` | scalar | float | Arc length in model units |
| `dihedral_angle` | scalar | float | Angle between adjacent face normals (radians) |
| `convexity` | scalar | int | 0=concave, 1=convex, 2=tangent |

### Curve Type IDs
| ID | Type |
|----|------|
| 0 | Line |
| 1 | Circle |
| 2 | Ellipse |
| 3 | Hyperbola |
| 4 | Parabola |
| 5 | Bezier Curve |
| 6 | B-Spline Curve |
| 7 | Other |

### Curve Grid Structure
Each `curve_grid[p]` contains 6 values:
```
[x, y, z, tx, ty, tz]
```
- `x, y, z`: 3D point on curve
- `tx, ty, tz`: Tangent direction at point

---

## Global Encodings

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `spatial_pos` | `(N, N)` | int32 | Shortest path distance between faces |
| `edge_path` | `(N, N, 16)` | int32 | Edge indices along shortest path (-1 = padding) |
| `d2_distance` | `(N, N, 64)` | int32 | D2 shape distribution histogram (pairwise) |
| `angle_distance` | `(N, N, 64)` | int32 | A3 angular distribution histogram (global, broadcast) |

---

## Python Interface

```python
from analysis_engine.brep_graph import BRepGraph, BRepGraphGenerator

# Generate graph from STEP file
generator = BRepGraphGenerator()
graph: BRepGraph = generator.generate("part.step")

# Access node features
for face_id, face_data in graph.faces.items():
    print(f"Face {face_id}: type={face_data['surface_type']}, area={face_data['area']}")

# Access edge features
for edge_id, edge_data in graph.edges.items():
    print(f"Edge {edge_id}: convexity={edge_data['convexity']}")

# Access global encodings
print(f"Spatial positions: {graph.spatial_pos.shape}")
```

---

## Determinism Guarantees

1. **Seed Control**: `np.random.seed(42)` ensures reproducible sampling
2. **Enumeration Order**: Faces/edges enumerated via deterministic OCP traversal
3. **Hash Stability**: Same STEP file produces identical graph across runs

---

## BrepMFR Compatibility

> [!NOTE]
> This schema is designed to be compatible with BrepMFR graph neural network inputs.
> However, BrepMFR model inference is **NOT** currently available due to missing
> pretrained checkpoints. The schema can be consumed by any GNN or rule-based system.
