import sys
import os
import numpy as np
import math
import queue
import bisect
import time

# OCP Imports
from OCP.STEPControl import STEPControl_Reader
from OCP.TopoDS import TopoDS
from OCP.TopExp import TopExp_Explorer, TopExp
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_IN, TopAbs_ON, TopAbs_FORWARD, TopAbs_REVERSED
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
    GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve
)
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRep import BRep_Tool
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.gp import gp_Pnt, gp_Vec, gp_Pnt2d
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape


def analyze_step_faces(step_path: str) -> dict:
    """
    Loads a STEP file and constructs a face adjacency graph with exact feature extraction.
    Steps 1, 2, 3: Full Pipeline (OCP Port - Final Iteration Fix).
    
    Args:
        step_path: Path to the STEP file.
        
    Returns:
        Dictionary containing face features, edge features, and Global Tensors:
        - spatial_pos: (N, N)
        - edge_path: (N, N, max_dist)
        - d2_distance: (N, N, 64)
        - angle_distance: (N, N, 64)
    """
    np.random.seed(42) # Determinism
    
    if not os.path.exists(step_path):
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    # 1. Load STEP file
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_path)
    
    if status != IFSelect_RetDone:
        raise ValueError(f"Error reading STEP file: {step_path}")
        
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    # 2. Face Enumeration & Identity
    all_faces = []
    # OCP TopExp_Explorer is NOT iterable in Python. Must use More()/Next().
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        all_faces.append(face)
        explorer.Next()
        
    num_faces = len(all_faces)
    
    # 3. Adjacency Construction & Edge Enumeration
    adj_list = {i: [] for i in range(num_faces)}
    all_graph_edges = [] 
    face_pair_to_edge_idx = {}
    
    # Use OCP Map for Shape->Ancestors
    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_map)
    n_edges = edge_map.Extent()
    
    for i in range(1, n_edges + 1):
        # Retrieve Edge and Neighbors
        edge = TopoDS.Edge_s(edge_map.FindKey(i))
        faces_list = edge_map.FindFromIndex(i)
        
        # Check if 2 faces
        if faces_list.Extent() == 2:
            # faces_list is TopTools_ListOfShape, which IS iterable in OCP/Python.
            f_objs = []
            # faces_list is TopTools_ListOfShape, which IS python-iterable (has __iter__)
            f_objs = []
            for f_shape in faces_list:
                f_objs.append(TopoDS.Face_s(f_shape))
            
            f1, f2 = f_objs[0], f_objs[1]
            
            # Find ID
            id1 = -1
            id2 = -1
            
            # Linear scan (Safe, Deterministic)
            found_cnt = 0
            for idx, known_face in enumerate(all_faces):
                if known_face.IsSame(f1):
                    id1 = idx
                    found_cnt += 1
                elif known_face.IsSame(f2):
                    id2 = idx
                    found_cnt += 1
                if found_cnt == 2:
                    break
            
            if id1 != -1 and id2 != -1 and id1 != id2:
                edge_idx = len(all_graph_edges)
                all_graph_edges.append((edge, id1, id2))
                
                # Bi-directional mapping for path lookup
                face_pair_to_edge_idx[(id1, id2)] = edge_idx
                face_pair_to_edge_idx[(id2, id1)] = edge_idx
                
                if id2 not in adj_list[id1]:
                    adj_list[id1].append(id2)
                if id1 not in adj_list[id2]:
                    adj_list[id2].append(id1)

    # 4. Face Processing & Data Collection
    faces_data = {}
    face_areas = []
    face_bounds = [] 
    
    for f_id, face in enumerate(all_faces):
        surf_adaptor = BRepAdaptor_Surface(face)
        occ_type = surf_adaptor.GetType()
        surface_type_int = 7
        
        if occ_type == GeomAbs_Plane: surface_type_int = 0
        elif occ_type == GeomAbs_Cylinder: surface_type_int = 1
        elif occ_type == GeomAbs_Cone: surface_type_int = 2
        elif occ_type == GeomAbs_Sphere: surface_type_int = 3
        elif occ_type == GeomAbs_Torus: surface_type_int = 4
        elif occ_type == GeomAbs_BezierSurface: surface_type_int = 5
        elif occ_type == GeomAbs_BSplineSurface: surface_type_int = 6
        
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = props.Mass()
        face_areas.append(area)
        
        # TopExp_Explorer is NOT iterable
        w_exp = TopExp_Explorer(face, TopAbs_WIRE)
        loop_count = 0
        while w_exp.More():
            loop_count += 1
            w_exp.Next()
            
        u_min = surf_adaptor.FirstUParameter()
        u_max = surf_adaptor.LastUParameter()
        v_min = surf_adaptor.FirstVParameter()
        v_max = surf_adaptor.LastVParameter()
        face_bounds.append((u_min, u_max, v_min, v_max))
        
        # UV Grid (5x5)
        u_grid = np.linspace(u_min, u_max, 5)
        v_grid = np.linspace(v_min, v_max, 5)
        grid_data = np.zeros((5, 5, 7), dtype=np.float32)
        
        # Init with dummy point to load face (since Load() is missing)
        classifier = BRepClass_FaceClassifier(face, gp_Pnt2d(0,0), 1e-7)
        # classifier.Load(face)  # Removed for OCP partial binding compatibility
        
        for i_u, u in enumerate(u_grid):
            for j_v, v in enumerate(v_grid):
                pnt = gp_Pnt()
                d1u = gp_Vec()
                d1v = gp_Vec()
                surf_adaptor.D1(u, v, pnt, d1u, d1v)
                
                norm_vec = d1u.Crossed(d1v)
                if norm_vec.Magnitude() < 1e-9:
                    nx, ny, nz = 0.0, 0.0, 0.0
                else:
                    norm_vec.Normalize()
                    nx, ny, nz = norm_vec.X(), norm_vec.Y(), norm_vec.Z()
                
                # Perform requires (Face, Pnt2d, Tol) in this binding
                classifier.Perform(face, gp_Pnt2d(u, v), 1e-7)
                state = classifier.State()
                t_val = 1.0 if (state == TopAbs_IN or state == TopAbs_ON) else 0.0
                
                grid_data[i_u, j_v] = [pnt.X(), pnt.Y(), pnt.Z(), nx, ny, nz, t_val]
        
        # --- DIMENSIONAL PARAMETERS (New Schema Extension) ---
        geom_params = {}
        
        if occ_type == GeomAbs_Plane:
            plane = surf_adaptor.Plane()
            loc = plane.Location()
            axis = plane.Axis().Direction()
            geom_params = {
                "origin": [loc.X(), loc.Y(), loc.Z()],
                "normal": [axis.X(), axis.Y(), axis.Z()]
            }
        elif occ_type == GeomAbs_Cylinder:
            cyl = surf_adaptor.Cylinder()
            loc = cyl.Location()
            axis = cyl.Axis().Direction()
            geom_params = {
                "radius": cyl.Radius(),
                "diameter": 2 * cyl.Radius(),
                "axis_origin": [loc.X(), loc.Y(), loc.Z()],
                "axis_direction": [axis.X(), axis.Y(), axis.Z()]
            }
        elif occ_type == GeomAbs_Cone:
            cone = surf_adaptor.Cone()
            apex = cone.Apex()
            axis = cone.Axis().Direction()
            geom_params = {
                "half_angle": cone.SemiAngle(),
                "apex": [apex.X(), apex.Y(), apex.Z()],
                "axis_direction": [axis.X(), axis.Y(), axis.Z()]
            }
        elif occ_type == GeomAbs_Sphere:
            sphere = surf_adaptor.Sphere()
            center = sphere.Location()
            geom_params = {
                "radius": sphere.Radius(),
                "center": [center.X(), center.Y(), center.Z()]
            }
        elif occ_type == GeomAbs_Torus:
            torus = surf_adaptor.Torus()
            center = torus.Location()
            axis = torus.Axis().Direction()
            geom_params = {
                "major_radius": torus.MajorRadius(),
                "minor_radius": torus.MinorRadius(),
                "center": [center.X(), center.Y(), center.Z()],
                "axis_direction": [axis.X(), axis.Y(), axis.Z()]
            }
        # BSpline/Bezier: No simple params, leave empty
        
        faces_data[f_id] = {
            "surface_type": surface_type_int,
            "area": float(area),
            "loop_count": loop_count,
            "adjacent_faces": sorted(list(set(adj_list[f_id]))),
            "adjacency_count": len(adj_list[f_id]),
            "uv_grid": grid_data,
            "geom_params": geom_params  # NEW: Dimensional parameters
        }


    # 5. Edge Processing & Raw Feature Extraction
    edges_data = {}
    
    for e_id, (edge, id1, id2) in enumerate(all_graph_edges):
        curve_adaptor = BRepAdaptor_Curve(edge)
        
        # --- Type ---
        c_type = curve_adaptor.GetType()
        curve_type_int = 7
        if c_type == GeomAbs_Line: curve_type_int = 0
        elif c_type == GeomAbs_Circle: curve_type_int = 1
        elif c_type == GeomAbs_Ellipse: curve_type_int = 2
        elif c_type == GeomAbs_Hyperbola: curve_type_int = 3
        elif c_type == GeomAbs_Parabola: curve_type_int = 4
        elif c_type == GeomAbs_BezierCurve: curve_type_int = 5
        elif c_type == GeomAbs_BSplineCurve: curve_type_int = 6
        
        # --- Length ---
        length = GCPnts_AbscissaPoint.Length_s(curve_adaptor)
        
        # --- Grid ---
        first = curve_adaptor.FirstParameter()
        last = curve_adaptor.LastParameter()
        params = np.linspace(first, last, 5)
        curve_grid = np.zeros((5, 6), dtype=np.float32)
        
        for i_p, param in enumerate(params):
            pnt = gp_Pnt()
            vec = gp_Vec()
            curve_adaptor.D1(param, pnt, vec) 
            
            tx, ty, tz = vec.X(), vec.Y(), vec.Z()
            if vec.Magnitude() > 1e-9:
                vec.Normalize()
                tx, ty, tz = vec.X(), vec.Y(), vec.Z()
            else:
                tx, ty, tz = 0.0, 0.0, 0.0
            curve_grid[i_p] = [pnt.X(), pnt.Y(), pnt.Z(), tx, ty, tz]
            
        # --- Convexity & Angle ---
        mid_param = (first + last) / 2.0
        p_mid = gp_Pnt()
        t_mid = gp_Vec()
        curve_adaptor.D1(mid_param, p_mid, t_mid)
        if t_mid.Magnitude() > 1e-9: t_mid.Normalize()
        
        f1_obj = all_faces[id1]
        c2d_1 = BRep_Tool.CurveOnSurface_s(edge, f1_obj, 0.0, 0.0)
        f1_p, l1_p = BRep_Tool.Range_s(edge, f1_obj)
        uv1 = c2d_1.Value(mid_param)
        surf1 = BRepAdaptor_Surface(f1_obj)
        dummy_p = gp_Pnt()
        d1u_1 = gp_Vec()
        d1v_1 = gp_Vec()
        surf1.D1(uv1.X(), uv1.Y(), dummy_p, d1u_1, d1v_1)
        n1 = d1u_1.Crossed(d1v_1)
        if n1.Magnitude() > 1e-9: n1.Normalize()
        if f1_obj.Orientation() == TopAbs_REVERSED: n1.Reverse()

        f2_obj = all_faces[id2]
        c2d_2 = BRep_Tool.CurveOnSurface_s(edge, f2_obj, 0.0, 0.0)
        f2_p, l2_p = BRep_Tool.Range_s(edge, f2_obj)
        uv2 = c2d_2.Value(mid_param)
        surf2 = BRepAdaptor_Surface(f2_obj)
        dummy_p2 = gp_Pnt()
        d1u_2 = gp_Vec()
        d1v_2 = gp_Vec()
        surf2.D1(uv2.X(), uv2.Y(), dummy_p2, d1u_2, d1v_2)
        n2 = d1u_2.Crossed(d1v_2)
        if n2.Magnitude() > 1e-9: n2.Normalize()
        if f2_obj.Orientation() == TopAbs_REVERSED: n2.Reverse()

        dot_prod = n1.Dot(n2)
        if dot_prod > 1.0: dot_prod = 1.0
        if dot_prod < -1.0: dot_prod = -1.0
        angle = math.acos(dot_prod)
        
        convexity = 2 
        if angle < 1e-2:
            convexity = 2
        else:
            is_f1_forward = False
            # TopExp_Explorer is NOT iterable
            ex = TopExp_Explorer(f1_obj, TopAbs_EDGE)
            while ex.More():
                curr_edge = TopoDS.Edge_s(ex.Current())
                if curr_edge.IsSame(edge):
                    if curr_edge.Orientation() == TopAbs_FORWARD:
                        is_f1_forward = True
                    break
                ex.Next()
            
            if is_f1_forward:
                n_left = n1
                n_right = n2
            else:
                n_left = n2
                n_right = n1
            
            cross_prod = n_left.Crossed(n_right)
            sign_val = cross_prod.Dot(t_mid)
            convexity = 1 if sign_val > 0 else 0
                    
        edges_data[e_id] = {
            "face_ids": (id1, id2),
            "curve_type": curve_type_int,
            "length": float(length),
            "dihedral_angle": float(angle),
            "convexity": int(convexity),
            "curve_grid": curve_grid
        }

    # 6. Global Matrices
    
    # --- spatial_pos (A1) & edge_path ---
    spatial_pos = np.full((num_faces, num_faces), 256, dtype=np.int32)
    max_dist = 16
    edge_path = np.full((num_faces, num_faces, max_dist), -1, dtype=np.int32)
    
    for start_node in range(num_faces):
        spatial_pos[start_node, start_node] = 0
        q = queue.Queue()
        q.put(start_node)
        pred = {start_node: None}
        visited = {start_node}
        
        while not q.empty():
            u = q.get()
            dist_u = spatial_pos[start_node, u]
            if dist_u >= 256: continue
                
            for v in adj_list[u]:
                if v not in visited:
                    visited.add(v)
                    spatial_pos[start_node, v] = dist_u + 1
                    edge_idx = face_pair_to_edge_idx.get((u, v), -1)
                    pred[v] = (u, edge_idx)
                    q.put(v)
                    
        for target_node in range(num_faces):
            if target_node in pred and target_node != start_node:
                curr = target_node
                path = []
                while curr != start_node:
                    parent, e_idx = pred[curr]
                    path.append(e_idx)
                    curr = parent
                
                path.reverse()
                
                # Truncate to max_dist
                path_len = min(len(path), max_dist)
                edge_path[start_node, target_node, :path_len] = path[:path_len]

    # --- Shape Distributions (d2_distance, angle_distance) ---
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    diag = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
    if diag < 1e-6: diag = 1.0
    
    d2_distance = np.zeros((num_faces, num_faces, 64), dtype=np.int32)
    angle_distance = np.zeros((num_faces, num_faces, 64), dtype=np.int32)
    
    classifiers_list = []
    for f in all_faces:
        c = BRepClass_FaceClassifier(f, gp_Pnt2d(0,0), 1e-7)
        classifiers_list.append(c)
    
    def sample_point_on_face(f_idx):
        face = all_faces[f_idx]
        surf = BRepAdaptor_Surface(face)
        u_min, u_max, v_min, v_max = face_bounds[f_idx]
        cls = classifiers_list[f_idx]
        
        for _ in range(100): 
            u = np.random.uniform(u_min, u_max)
            v = np.random.uniform(v_min, v_max)
            cls.Perform(face, gp_Pnt2d(u, v), 1e-7)
            if cls.State() in [TopAbs_IN, TopAbs_ON]:
                p = surf.Value(u, v)
                return np.array([p.X(), p.Y(), p.Z()])
        return np.array([0., 0., 0.])

    # Global A3 Sampling
    total_area = sum(face_areas)
    if total_area < 1e-9:
        cdf = np.linspace(0, 1, num_faces+1)[1:]
    else:
        cdf = np.cumsum(face_areas) / total_area
        
    def sample_global_point():
        r = np.random.rand()
        f_idx = bisect.bisect_right(cdf, r)
        if f_idx >= num_faces: f_idx = num_faces - 1
        return sample_point_on_face(f_idx)
    
    # Compute A3 Global Histogram
    a3_vals = []
    for _ in range(512):
        p1 = sample_global_point()
        p2 = sample_global_point()
        p3 = sample_global_point()
        
        v1 = p1 - p2
        v2 = p3 - p2
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-6 or n2 < 1e-6:
            angle = 0.0
        else:
            dot = np.dot(v1, v2) / (n1 * n2)
            dot = np.clip(dot, -1.0, 1.0)
            angle = math.acos(dot)
        a3_vals.append(angle)
        
    a3_arr = np.array(a3_vals) / np.pi
    a3_arr = np.clip(a3_arr, 0.0, 0.999)
    bin_a3 = (a3_arr * 64).astype(np.int32)
    a3_global_hist = np.bincount(bin_a3, minlength=64)
    
    angle_distance[:] = a3_global_hist

    # Compute D2 (Local)
    for i in range(num_faces):
        for j in range(num_faces):
            # D2 Local
            area_i = face_areas[i]
            area_j = face_areas[j]
            t_area = area_i + area_j
            p_i = 0.5 if t_area < 1e-9 else area_i / t_area
            
            d2_vals = []
            for _ in range(512):
                idx1 = i if np.random.rand() < p_i else j
                p1 = sample_point_on_face(idx1)
                idx2 = i if np.random.rand() < p_i else j
                p2 = sample_point_on_face(idx2)
                dist = np.linalg.norm(p1 - p2)
                d2_vals.append(dist)
                
            d2_arr = np.array(d2_vals) / diag
            d2_arr = np.clip(d2_arr, 0.0, 0.999)
            bin_d2 = (d2_arr * 64).astype(np.int32)
            d2_hist = np.bincount(bin_d2, minlength=64)
            
            d2_distance[i, j] = d2_hist

    return {
        "faces": faces_data, 
        "edges": edges_data,
        "spatial_pos": spatial_pos,
        "edge_path": edge_path,
        "d2_distance": d2_distance,
        "angle_distance": angle_distance
    }


def analyze_step_faces_fast(
    step_path: str,
    time_budget_sec: float = 5.0,
    max_faces: int = 120,
    rep_points_per_face: int = 12,
    global_a3_samples: int = 64
) -> dict:
    """
    Fast-path B-Rep graph generator for production inference.
    Returns the same output keys and tensor shapes as analyze_step_faces,
    but with bounded, approximate computation suitable for real-time inference.
    
    Args:
        step_path: Path to the STEP file.
        time_budget_sec: Maximum wall-clock time allowed.
        max_faces: If face count exceeds this, restrict D2 to adjacency radius ≤ 2.
        rep_points_per_face: Number of representative points per face for D2/A3.
        global_a3_samples: Number of triplets for global A3 histogram.
        
    Returns:
        Dictionary with keys: faces, edges, spatial_pos, edge_path, d2_distance, angle_distance
    """
    start_time = time.time()
    np.random.seed(42)  # Determinism
    
    if not os.path.exists(step_path):
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    # =========================================================================
    # STEP 1: Load STEP file (verbatim from training-grade)
    # =========================================================================
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_path)
    
    if status != IFSelect_RetDone:
        raise ValueError(f"Error reading STEP file: {step_path}")
        
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    
    # =========================================================================
    # STEP 2: Face Enumeration & Identity (verbatim)
    # =========================================================================
    all_faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        all_faces.append(face)
        explorer.Next()
        
    num_faces = len(all_faces)
    
    # =========================================================================
    # Adjacency Construction & Edge Enumeration (verbatim)
    # =========================================================================
    adj_list = {i: [] for i in range(num_faces)}
    all_graph_edges = [] 
    face_pair_to_edge_idx = {}
    
    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_map)
    n_edges = edge_map.Extent()
    
    for i in range(1, n_edges + 1):
        edge = TopoDS.Edge_s(edge_map.FindKey(i))
        faces_list = edge_map.FindFromIndex(i)
        
        if faces_list.Extent() == 2:
            f_objs = []
            for f_shape in faces_list:
                f_objs.append(TopoDS.Face_s(f_shape))
            
            f1, f2 = f_objs[0], f_objs[1]
            
            id1 = -1
            id2 = -1
            found_cnt = 0
            for idx, known_face in enumerate(all_faces):
                if known_face.IsSame(f1):
                    id1 = idx
                    found_cnt += 1
                elif known_face.IsSame(f2):
                    id2 = idx
                    found_cnt += 1
                if found_cnt == 2:
                    break
            
            if id1 != -1 and id2 != -1 and id1 != id2:
                edge_idx = len(all_graph_edges)
                all_graph_edges.append((edge, id1, id2))
                face_pair_to_edge_idx[(id1, id2)] = edge_idx
                face_pair_to_edge_idx[(id2, id1)] = edge_idx
                
                if id2 not in adj_list[id1]:
                    adj_list[id1].append(id2)
                if id1 not in adj_list[id2]:
                    adj_list[id2].append(id1)

    # =========================================================================
    # Face Processing & Data Collection (verbatim - UV grid sampling)
    # =========================================================================
    faces_data = {}
    face_areas = []
    face_bounds = []
    face_rep_points = []  # NEW: Representative points for fast D2/A3
    
    for f_id, face in enumerate(all_faces):
        surf_adaptor = BRepAdaptor_Surface(face)
        occ_type = surf_adaptor.GetType()
        surface_type_int = 7
        
        if occ_type == GeomAbs_Plane: surface_type_int = 0
        elif occ_type == GeomAbs_Cylinder: surface_type_int = 1
        elif occ_type == GeomAbs_Cone: surface_type_int = 2
        elif occ_type == GeomAbs_Sphere: surface_type_int = 3
        elif occ_type == GeomAbs_Torus: surface_type_int = 4
        elif occ_type == GeomAbs_BezierSurface: surface_type_int = 5
        elif occ_type == GeomAbs_BSplineSurface: surface_type_int = 6
        
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = props.Mass()
        face_areas.append(area)
        
        w_exp = TopExp_Explorer(face, TopAbs_WIRE)
        loop_count = 0
        while w_exp.More():
            loop_count += 1
            w_exp.Next()
            
        u_min = surf_adaptor.FirstUParameter()
        u_max = surf_adaptor.LastUParameter()
        v_min = surf_adaptor.FirstVParameter()
        v_max = surf_adaptor.LastVParameter()
        face_bounds.append((u_min, u_max, v_min, v_max))
        
        # UV Grid (5x5) - verbatim
        u_grid = np.linspace(u_min, u_max, 5)
        v_grid = np.linspace(v_min, v_max, 5)
        grid_data = np.zeros((5, 5, 7), dtype=np.float32)
        
        classifier = BRepClass_FaceClassifier(face, gp_Pnt2d(0,0), 1e-7)
        
        for i_u, u in enumerate(u_grid):
            for j_v, v in enumerate(v_grid):
                pnt = gp_Pnt()
                d1u = gp_Vec()
                d1v = gp_Vec()
                surf_adaptor.D1(u, v, pnt, d1u, d1v)
                
                norm_vec = d1u.Crossed(d1v)
                if norm_vec.Magnitude() < 1e-9:
                    nx, ny, nz = 0.0, 0.0, 0.0
                else:
                    norm_vec.Normalize()
                    nx, ny, nz = norm_vec.X(), norm_vec.Y(), norm_vec.Z()
                
                classifier.Perform(face, gp_Pnt2d(u, v), 1e-7)
                state = classifier.State()
                t_val = 1.0 if (state == TopAbs_IN or state == TopAbs_ON) else 0.0
                
                grid_data[i_u, j_v] = [pnt.X(), pnt.Y(), pnt.Z(), nx, ny, nz, t_val]
        
        # NEW: Precompute representative points using stratified UV grid (no rejection sampling)
        rep_grid_size = int(math.ceil(math.sqrt(rep_points_per_face)))
        rep_u_vals = np.linspace(u_min, u_max, rep_grid_size + 2)[1:-1]
        rep_v_vals = np.linspace(v_min, v_max, rep_grid_size + 2)[1:-1]
        rep_pts = []
        for ru in rep_u_vals:
            for rv in rep_v_vals:
                if len(rep_pts) >= rep_points_per_face:
                    break
                p = surf_adaptor.Value(ru, rv)
                rep_pts.append(np.array([p.X(), p.Y(), p.Z()]))
            if len(rep_pts) >= rep_points_per_face:
                break
        # Pad if needed
        while len(rep_pts) < rep_points_per_face:
            p = surf_adaptor.Value((u_min + u_max) / 2, (v_min + v_max) / 2)
            rep_pts.append(np.array([p.X(), p.Y(), p.Z()]))
        face_rep_points.append(np.array(rep_pts[:rep_points_per_face]))
        
        # --- DIMENSIONAL PARAMETERS (Same as full path) ---
        geom_params = {}
        
        if occ_type == GeomAbs_Plane:
            plane = surf_adaptor.Plane()
            loc = plane.Location()
            axis = plane.Axis().Direction()
            geom_params = {
                "origin": [loc.X(), loc.Y(), loc.Z()],
                "normal": [axis.X(), axis.Y(), axis.Z()]
            }
        elif occ_type == GeomAbs_Cylinder:
            cyl = surf_adaptor.Cylinder()
            loc = cyl.Location()
            axis = cyl.Axis().Direction()
            geom_params = {
                "radius": cyl.Radius(),
                "diameter": 2 * cyl.Radius(),
                "axis_origin": [loc.X(), loc.Y(), loc.Z()],
                "axis_direction": [axis.X(), axis.Y(), axis.Z()]
            }
        elif occ_type == GeomAbs_Cone:
            cone = surf_adaptor.Cone()
            apex = cone.Apex()
            axis = cone.Axis().Direction()
            geom_params = {
                "half_angle": cone.SemiAngle(),
                "apex": [apex.X(), apex.Y(), apex.Z()],
                "axis_direction": [axis.X(), axis.Y(), axis.Z()]
            }
        elif occ_type == GeomAbs_Sphere:
            sphere = surf_adaptor.Sphere()
            center = sphere.Location()
            geom_params = {
                "radius": sphere.Radius(),
                "center": [center.X(), center.Y(), center.Z()]
            }
        elif occ_type == GeomAbs_Torus:
            torus = surf_adaptor.Torus()
            center = torus.Location()
            axis = torus.Axis().Direction()
            geom_params = {
                "major_radius": torus.MajorRadius(),
                "minor_radius": torus.MinorRadius(),
                "center": [center.X(), center.Y(), center.Z()],
                "axis_direction": [axis.X(), axis.Y(), axis.Z()]
            }
        
        faces_data[f_id] = {
            "surface_type": surface_type_int,
            "area": float(area),
            "loop_count": loop_count,
            "adjacent_faces": sorted(list(set(adj_list[f_id]))),
            "adjacency_count": len(adj_list[f_id]),
            "uv_grid": grid_data,
            "geom_params": geom_params
        }


    # =========================================================================
    # Edge Processing & Raw Feature Extraction (verbatim)
    # =========================================================================
    edges_data = {}
    
    for e_id, (edge, id1, id2) in enumerate(all_graph_edges):
        curve_adaptor = BRepAdaptor_Curve(edge)
        
        c_type = curve_adaptor.GetType()
        curve_type_int = 7
        if c_type == GeomAbs_Line: curve_type_int = 0
        elif c_type == GeomAbs_Circle: curve_type_int = 1
        elif c_type == GeomAbs_Ellipse: curve_type_int = 2
        elif c_type == GeomAbs_Hyperbola: curve_type_int = 3
        elif c_type == GeomAbs_Parabola: curve_type_int = 4
        elif c_type == GeomAbs_BezierCurve: curve_type_int = 5
        elif c_type == GeomAbs_BSplineCurve: curve_type_int = 6
        
        length = GCPnts_AbscissaPoint.Length_s(curve_adaptor)
        
        first = curve_adaptor.FirstParameter()
        last = curve_adaptor.LastParameter()
        params = np.linspace(first, last, 5)
        curve_grid = np.zeros((5, 6), dtype=np.float32)
        
        for i_p, param in enumerate(params):
            pnt = gp_Pnt()
            vec = gp_Vec()
            curve_adaptor.D1(param, pnt, vec) 
            
            tx, ty, tz = vec.X(), vec.Y(), vec.Z()
            if vec.Magnitude() > 1e-9:
                vec.Normalize()
                tx, ty, tz = vec.X(), vec.Y(), vec.Z()
            else:
                tx, ty, tz = 0.0, 0.0, 0.0
            curve_grid[i_p] = [pnt.X(), pnt.Y(), pnt.Z(), tx, ty, tz]
            
        mid_param = (first + last) / 2.0
        p_mid = gp_Pnt()
        t_mid = gp_Vec()
        curve_adaptor.D1(mid_param, p_mid, t_mid)
        if t_mid.Magnitude() > 1e-9: t_mid.Normalize()
        
        f1_obj = all_faces[id1]
        c2d_1 = BRep_Tool.CurveOnSurface_s(edge, f1_obj, 0.0, 0.0)
        f1_p, l1_p = BRep_Tool.Range_s(edge, f1_obj)
        uv1 = c2d_1.Value(mid_param)
        surf1 = BRepAdaptor_Surface(f1_obj)
        dummy_p = gp_Pnt()
        d1u_1 = gp_Vec()
        d1v_1 = gp_Vec()
        surf1.D1(uv1.X(), uv1.Y(), dummy_p, d1u_1, d1v_1)
        n1 = d1u_1.Crossed(d1v_1)
        if n1.Magnitude() > 1e-9: n1.Normalize()
        if f1_obj.Orientation() == TopAbs_REVERSED: n1.Reverse()

        f2_obj = all_faces[id2]
        c2d_2 = BRep_Tool.CurveOnSurface_s(edge, f2_obj, 0.0, 0.0)
        f2_p, l2_p = BRep_Tool.Range_s(edge, f2_obj)
        uv2 = c2d_2.Value(mid_param)
        surf2 = BRepAdaptor_Surface(f2_obj)
        dummy_p2 = gp_Pnt()
        d1u_2 = gp_Vec()
        d1v_2 = gp_Vec()
        surf2.D1(uv2.X(), uv2.Y(), dummy_p2, d1u_2, d1v_2)
        n2 = d1u_2.Crossed(d1v_2)
        if n2.Magnitude() > 1e-9: n2.Normalize()
        if f2_obj.Orientation() == TopAbs_REVERSED: n2.Reverse()

        dot_prod = n1.Dot(n2)
        if dot_prod > 1.0: dot_prod = 1.0
        if dot_prod < -1.0: dot_prod = -1.0
        angle = math.acos(dot_prod)
        
        convexity = 2 
        if angle < 1e-2:
            convexity = 2
        else:
            is_f1_forward = False
            ex = TopExp_Explorer(f1_obj, TopAbs_EDGE)
            while ex.More():
                curr_edge = TopoDS.Edge_s(ex.Current())
                if curr_edge.IsSame(edge):
                    if curr_edge.Orientation() == TopAbs_FORWARD:
                        is_f1_forward = True
                    break
                ex.Next()
            
            if is_f1_forward:
                n_left = n1
                n_right = n2
            else:
                n_left = n2
                n_right = n1
            
            cross_prod = n_left.Crossed(n_right)
            sign_val = cross_prod.Dot(t_mid)
            convexity = 1 if sign_val > 0 else 0
                    
        edges_data[e_id] = {
            "face_ids": (id1, id2),
            "curve_type": curve_type_int,
            "length": float(length),
            "dihedral_angle": float(angle),
            "convexity": int(convexity),
            "curve_grid": curve_grid
        }

    # =========================================================================
    # A1 (spatial_pos) & edge_path - verbatim (already fast: O(N*E))
    # =========================================================================
    spatial_pos = np.full((num_faces, num_faces), 256, dtype=np.int32)
    max_dist = 16
    edge_path = np.full((num_faces, num_faces, max_dist), -1, dtype=np.int32)
    
    for start_node in range(num_faces):
        spatial_pos[start_node, start_node] = 0
        q = queue.Queue()
        q.put(start_node)
        pred = {start_node: None}
        visited = {start_node}
        
        while not q.empty():
            u = q.get()
            dist_u = spatial_pos[start_node, u]
            if dist_u >= 256: continue
                
            for v in adj_list[u]:
                if v not in visited:
                    visited.add(v)
                    spatial_pos[start_node, v] = dist_u + 1
                    edge_idx = face_pair_to_edge_idx.get((u, v), -1)
                    pred[v] = (u, edge_idx)
                    q.put(v)
                    
        for target_node in range(num_faces):
            if target_node in pred and target_node != start_node:
                curr = target_node
                path = []
                while curr != start_node:
                    parent, e_idx = pred[curr]
                    path.append(e_idx)
                    curr = parent
                
                path.reverse()
                path_len = min(len(path), max_dist)
                edge_path[start_node, target_node, :path_len] = path[:path_len]

    # =========================================================================
    # FAST STEP 3: Bounded Shape Distributions
    # =========================================================================
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    diag = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
    if diag < 1e-6: diag = 1.0
    
    d2_distance = np.zeros((num_faces, num_faces, 64), dtype=np.int32)
    angle_distance = np.zeros((num_faces, num_faces, 64), dtype=np.int32)
    
    # -------------------------------------------------------------------------
    # 2.3: Fast Global A3 (O(1) - uses precomputed representative points)
    # -------------------------------------------------------------------------
    all_rep_points_flat = np.vstack(face_rep_points)  # (N * rep_points_per_face, 3)
    n_total_rep = all_rep_points_flat.shape[0]
    
    a3_vals = []
    for _ in range(global_a3_samples):
        idx1, idx2, idx3 = np.random.randint(0, n_total_rep, 3)
        p1 = all_rep_points_flat[idx1]
        p2 = all_rep_points_flat[idx2]
        p3 = all_rep_points_flat[idx3]
        
        v1 = p1 - p2
        v2 = p3 - p2
        n1_norm = np.linalg.norm(v1)
        n2_norm = np.linalg.norm(v2)
        
        if n1_norm < 1e-6 or n2_norm < 1e-6:
            angle = 0.0
        else:
            dot = np.dot(v1, v2) / (n1_norm * n2_norm)
            dot = np.clip(dot, -1.0, 1.0)
            angle = math.acos(dot)
        a3_vals.append(angle)
    
    a3_arr = np.array(a3_vals) / np.pi
    a3_arr = np.clip(a3_arr, 0.0, 0.999)
    bin_a3 = (a3_arr * 64).astype(np.int32)
    a3_global_hist = np.bincount(bin_a3, minlength=64)[:64]
    
    # Broadcast to all pairs
    angle_distance[:] = a3_global_hist
    
    # -------------------------------------------------------------------------
    # 2.2: Fast D2 approximation (bounded)
    # -------------------------------------------------------------------------
    # Determine which pairs to compute
    use_sparse_d2 = num_faces > max_faces
    
    # Compute adjacency radius 2 set if needed
    adj_radius_2 = {}
    if use_sparse_d2:
        for i in range(num_faces):
            neighbors = set(adj_list[i])
            neighbors_r2 = set()
            for n1 in adj_list[i]:
                for n2 in adj_list[n1]:
                    neighbors_r2.add(n2)
            neighbors_r2.update(neighbors)
            neighbors_r2.discard(i)
            adj_radius_2[i] = neighbors_r2
    
    # Neutral histogram for skipped pairs
    neutral_hist = np.zeros(64, dtype=np.int32)
    neutral_hist[32] = 16  # Centered mass
    
    d2_samples_per_pair = 24  # Fixed small sample count
    
    elapsed = time.time() - start_time
    budget_remaining = time_budget_sec - elapsed
    
    for i in range(num_faces):
        # Time budget check
        if time.time() - start_time > time_budget_sec:
            # Fill remaining with neutral
            for ii in range(i, num_faces):
                for jj in range(num_faces):
                    d2_distance[ii, jj] = neutral_hist
            break
            
        for j in range(num_faces):
            # Skip if using sparse mode and pair is not within radius 2
            if use_sparse_d2 and j not in adj_radius_2.get(i, set()) and i != j:
                d2_distance[i, j] = neutral_hist
                continue
            
            # Fast D2: sample from precomputed representative points
            pts_i = face_rep_points[i]  # (rep_points_per_face, 3)
            pts_j = face_rep_points[j]  # (rep_points_per_face, 3)
            
            # Sample indices
            idx_i = np.random.randint(0, rep_points_per_face, d2_samples_per_pair)
            idx_j = np.random.randint(0, rep_points_per_face, d2_samples_per_pair)
            
            # Compute distances
            dists = np.linalg.norm(pts_i[idx_i] - pts_j[idx_j], axis=1)
            
            # Histogram
            d2_arr = dists / diag
            d2_arr = np.clip(d2_arr, 0.0, 0.999)
            bin_d2 = (d2_arr * 64).astype(np.int32)
            d2_hist = np.bincount(bin_d2, minlength=64)[:64]
            
            d2_distance[i, j] = d2_hist

    elapsed_final = time.time() - start_time
    print(f"[FAST] Faces: {num_faces}, Time: {elapsed_final:.3f}s")
    
    return {
        "faces": faces_data, 
        "edges": edges_data,
        "spatial_pos": spatial_pos,
        "edge_path": edge_path,
        "d2_distance": d2_distance,
        "angle_distance": angle_distance
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        step_filename = sys.argv[1]
    else:
        print("Please provide a STEP file path as argument.")
        sys.exit(1)
    
    # Check for --fast flag
    use_fast = "--fast" in sys.argv
    
    try:
        if use_fast:
            print("Using FAST path...")
            results = analyze_step_faces_fast(step_filename)
        else:
            print("Using TRAINING-GRADE path...")
            results = analyze_step_faces(step_filename)
        
        print(f"Num Faces: {len(results['faces'])}")
        print(f"Num Edges: {len(results['edges'])}")
        
        if len(results['faces']) > 0:
            print(f"Face 0 Grid Shape: {results['faces'][0]['uv_grid'].shape}")
        
        if len(results['edges']) > 0:
            print(f"Edge 0 Grid Shape: {results['edges'][0]['curve_grid'].shape}")
            
        print(f"spatial_pos Shape: {results['spatial_pos'].shape}")
        print(f"d2_distance Shape: {results['d2_distance'].shape}")
        print(f"angle_distance Shape: {results['angle_distance'].shape}")
        print(f"edge_path Shape: {results['edge_path'].shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
