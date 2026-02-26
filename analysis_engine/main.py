import argparse
import sys
import cadquery as cq
import traceback
from OCP.RWStl import RWStl

class GeometryContext:
    def __init__(self):
        self.step_solids = []
        self.step_faces = []
        self.step_face_ids = {} # {id: face}
        self.step_bbox = None
        self.mesh_object = None
        self.mesh_vertices_count = 0
        self.mesh_faces_count = 0
        self.mesh_bbox = None
        self.assumed_coordinate_frame = "Shared World Coordinate System"
        self.total_volume = 0.0 # Geometric Fact: Total Volume of STEP solids

def process_step_file(step_file_path):
    print(f"Attempting to load STEP file: {step_file_path}", flush=True)
    try:
        step_objects = cq.importers.importStep(step_file_path)
        
        step_solids = step_objects.solids().vals()
        step_faces = step_objects.faces().vals()
        
        print(f"STEP Loaded. Solids: {len(step_solids)}, Faces: {len(step_faces)}", flush=True)
        
        # Calculate Total Part Volume (Geometric Fact)
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        
        total_volume = 0.0
        vprops = GProp_GProps()
        for solid in step_solids:
            BRepGProp.VolumeProperties_s(solid.wrapped, vprops)
            total_volume += vprops.Mass()
            
        print(f"Total Part Volume (Geometric Fact): {total_volume:.2f} mm^3", flush=True)

        # Assign IDs
        step_face_ids = {i: f for i, f in enumerate(step_faces)}
        
        # Compute BBox
        min_pt = [float('inf')]*3
        max_pt = [float('-inf')]*3
        
        has_bbox = False
        for s in step_solids:
            try:
                b = s.BoundingBox()
                min_pt[0] = min(min_pt[0], b.xmin)
                min_pt[1] = min(min_pt[1], b.ymin)
                min_pt[2] = min(min_pt[2], b.zmin)
                max_pt[0] = max(max_pt[0], b.xmax)
                max_pt[1] = max(max_pt[1], b.ymax)
                max_pt[2] = max(max_pt[2], b.zmax)
                has_bbox = True
            except:
                pass
        
        step_bbox = (tuple(min_pt), tuple(max_pt)) if has_bbox else None
        
        return step_solids, step_faces, step_face_ids, step_bbox, total_volume

    except Exception as e:
        print(f"Error loading STEP: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def process_mesh_file(file_path):
    print("Attempting to load Mesh file...", flush=True)
    try:
         # Use OCP RWStl to read STL directly
         mesh = RWStl.ReadFile_s(file_path)
         if mesh is None:
             raise ValueError("Loaded mesh is None")

         print("Mesh file loaded successfully.", flush=True)
         
         # Count Vertices and Faces
         mesh_vert_count = mesh.NbNodes()
         mesh_face_count = mesh.NbTriangles()
         
         # Bounding Box
         min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
         max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
         
         # Iterate nodes to find bbox (1-based index)
         for i in range(1, mesh_vert_count + 1):
            pnt = mesh.Node(i)
            x, y, z = pnt.X(), pnt.Y(), pnt.Z()
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            if z < min_z: min_z = z
            if x > max_x: max_x = x
            if y > max_y: max_y = y
            if z > max_z: max_z = z
         
         print(f"Mesh Vertices: {mesh_vert_count}", flush=True)
         print(f"Mesh Faces: {mesh_face_count}", flush=True)
         
         if mesh_vert_count > 0:
            print(f"Mesh BBox: Min({min_x:.4f}, {min_y:.4f}, {min_z:.4f}) Max({max_x:.4f}, {max_y:.4f}, {max_z:.4f})", flush=True)
            bbox = (min_x, min_y, min_z, max_x, max_y, max_z)
         else:
            print("Mesh BBox: Empty", flush=True)
            bbox = None
            
         # Return data for context
         return mesh, mesh_vert_count, mesh_face_count, bbox
            
    except Exception:
         print("Error processing Mesh file", flush=True)
         raise

def print_backend_info():
    print("\n=== System Backend Info ===", flush=True)
    try:
        import OCP
        ocp_version = getattr(OCP, '__version__', 'Unknown')
        print(f"Python Binding: OCP (Open Cascade Python) {ocp_version}", flush=True)
        
        # Try to get OCCT version
        occt_version = None
        detection_source = None
        
        try:
            from OCP.Standard import Standard_Version
            occt_version = Standard_Version.String_s()
            detection_source = "Detected via API"
        except Exception:
            pass
            
        if not occt_version and ocp_version != 'Unknown':
            # Infer from OCP version (e.g., 7.7.2.0 -> 7.7.2)
            parts = ocp_version.split('.')
            if len(parts) >= 3:
                occt_version = ".".join(parts[:3])
                detection_source = "Inferred from Binding"
        
        if occt_version:
            print(f"Open Cascade (OCCT) Version: {occt_version} ({detection_source})", flush=True)
        else:
            print("Open Cascade (OCCT) Version: Detection Failed", flush=True)
            
    except ImportError:
        print("Python Binding: Unknown (OCP Import Failed)", flush=True)

    print("Backend Geometry Engine: Powered by native C++ code (Open Cascade Technology)", flush=True)
    print("=========================\n", flush=True)

def main():
    try:
        parser = argparse.ArgumentParser(description="Analysis Engine Entry Point")
        parser.add_argument("step_file_path", help="Path to the source STEP file")
        parser.add_argument("mesh_file_path", help="Path to the generated mesh file")

        args = parser.parse_args()

        print("Analysis Engine Started.", flush=True)
        print_backend_info() # Report Backend Details
        print(f"Received STEP file: {args.step_file_path}", flush=True)
        print(f"Received Mesh file: {args.mesh_file_path}", flush=True)

        geo_context = GeometryContext()

        # 1. Process STEP
        try:
            step_solids, step_faces, step_face_ids, step_bbox, total_volume = process_step_file(args.step_file_path)
            geo_context.step_solids = step_solids
            geo_context.step_faces = step_faces
            geo_context.step_face_ids = step_face_ids
            geo_context.step_bbox = step_bbox
            geo_context.total_volume = total_volume
        except Exception as e:
            print(f"STEP Processing Failed: {e}", flush=True)
            sys.exit(1)

        # 2. Process Mesh
        try:
            mesh_obj, vert_count, face_count, bbox = process_mesh_file(args.mesh_file_path)
            geo_context.mesh_object = mesh_obj
            geo_context.mesh_vertices_count = vert_count
            geo_context.mesh_faces_count = face_count
            geo_context.mesh_bbox = bbox
        except Exception as e:
            print(f"Mesh Processing Failed: {e}", flush=True)
            sys.exit(1)
            
        # Confirmation and Assumptions
        print(f"Geometry Context Created. STEP Faces: {len(geo_context.step_faces)}, Mesh Triangles: {geo_context.mesh_faces_count}", flush=True)
        
        print("\n=== Geometry Context Assumptions ===", flush=True)
        print(f"Coordinate Frame: {geo_context.assumed_coordinate_frame}", flush=True)
        
        sb = geo_context.step_bbox
        if sb:
            # sb is ((min_x, min_y, min_z), (max_x, max_y, max_z))
            print(f"STEP BBox Extents: Min({sb[0][0]:.4f}, {sb[0][1]:.4f}, {sb[0][2]:.4f}) Max({sb[1][0]:.4f}, {sb[1][1]:.4f}, {sb[1][2]:.4f})", flush=True)
        else:
             print("STEP BBox Extents: None", flush=True)
             
        mb = geo_context.mesh_bbox
        if mb:
            print(f"Mesh BBox Extents: Min({mb[0]:.4f}, {mb[1]:.4f}, {mb[2]:.4f}) Max({mb[3]:.4f}, {mb[4]:.4f}, {mb[5]:.4f})", flush=True)
        else:
            print("Mesh BBox Extents: None", flush=True)
            
        print("====================================", flush=True)

        print("\n=== Candidate Tool Axes ===", flush=True)
        candidates = []
        
        # 1. Principal Axes
        principals = [
            ((1, 0, 0), "Principal +X"),
            ((-1, 0, 0), "Principal -X"),
            ((0, 1, 0), "Principal +Y"),
            ((0, -1, 0), "Principal -Y"),
            ((0, 0, 1), "Principal +Z"),
            ((0, 0, -1), "Principal -Z")
        ]
        candidates.extend(principals)
        
        # 2. Planar Face Normals from STEP
        # Naive extraction - just unique normals from planar faces
        seen_normals = set()
        
        # Determine unique normals
        for face in geo_context.step_faces:
            try:
                # Check for Plane surface type using geomType
                if face.geomType() == "PLANE":
                   n = face.normalAt(None).toTuple() # Normalized tuple (x, y, z)
                   
                   # Rounding for simple de-duplication
                   rn = tuple(round(v, 4) for v in n)
                   
                   if rn not in seen_normals:
                       seen_normals.add(rn)
                       # Store slightly rounded or raw - raw is better for physics, rounded for display/dedup
                       # We'll re-normalize or just use 'n'
                       candidates.append( (n, "Planar Face Normal") )
            except:
                pass
                
        # Print Candidates
        for axis, source in candidates:
            # Format axis tuple
            ax_str = f"({axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f})"
            print(f"Axis: {ax_str} | Source: {source}", flush=True)
            
        print("===========================", flush=True)

        # 3. Accessibility Analysis (DeepMill with Fallback & Audit)
        from analysis_engine.deepmill import DeepMillAccessibilityEngine
        from analysis_engine.setup_context import SetupPartitioner
        
        print("\n=== Accessibility Analysis & Setup Partitioning (DeepMill Integrated) ===", flush=True)
        print(">>> DISCLAIMER: OUTPUT REPRESENTS GEOMETRIC FACTS ONLY. NO COSTING/PRICING IMPLICATIONS. <<<", flush=True)
        # Instantiate Real Engine (will verify dependencies internally)
        engine = DeepMillAccessibilityEngine()

        # --- B-Rep Graph Generation (Canonical Representation) ---
        print("\n=== Generating B-Rep Graph (Canonical Representation) ===", flush=True)
        brep_graph = None
        graph_recognizer = None
        try:
            from analysis_engine.brep_graph import BRepGraphGenerator, extract_subgraph
            from analysis_engine.brep_mfr import GraphNativeFeatureRecognizer
            
            graph_gen = BRepGraphGenerator(use_fast_path=True, time_budget_sec=5.0)
            brep_graph = graph_gen.generate(args.step_file_path)
            print(f"  {brep_graph.summary()}", flush=True)
            
            # Validate determinism
            is_det, det_msg = graph_gen.validate_determinism(args.step_file_path, runs=2)
            print(f"  Determinism Check: {det_msg}", flush=True)
            
            # Initialize graph-native feature recognizer
            graph_recognizer = GraphNativeFeatureRecognizer()
        except Exception as e:
            print(f"  B-Rep Graph Generation Failed: {e}", flush=True)
            print("  Continuing without graph-native features...", flush=True)
        
        # Hardcoded tool diameter for prototype (e.g., 6.0mm)
        tool_diameter = 6.0
        print(f"Tool Diameter: {tool_diameter}mm", flush=True)
        
        generated_setups = [] # Collect for comparison

        for axis, source in candidates:
            # Run Accessibility Query
            # The engine handles fallback and caching internally
            acc_result = engine.predict(
                geometry_context=geo_context,
                tool_axis=axis,
                tool_diameter=tool_diameter
            )
            print(acc_result.summary(), flush=True)
            
            # Extract Diagnostics if available (Subclass check)
            diagnostics = getattr(acc_result, 'deepmill_diagnostics', None)

            # Generate Setup Context (Partitioning) -> Determine Reachable Faces
            # We need to construct the setup first to know WHICH faces are reachable
            # But the constructor takes diagnostics. So current flow needs slight adjustment:
            # 1. Create Setup (DeepMill only first)
            # 2. Extract Reachable IDs
            # 3. Run BrepMFR on those IDs
            # 4. Attach to Setup (post-init or rebuild)

            # Optimizing: We can compute accessibility locally (DeepMill provides ratio, partitioner does geometric filtering)
            # Actually, `creates_setup_from_accessibility` DOES the filtering.
            # So we let it create the setup, THEN run BrepMFR, THEN attach.
            
            setup = SetupPartitioner.creates_setup_from_accessibility(
                geo_context=geo_context,
                axis=axis,
                tool_diameter=tool_diameter,
                accessibility_ratio=acc_result.accessibility_ratio,
                deepmill_diagnostics=diagnostics
            )
            
            # --- Graph-Native Feature Recognition (Diagnostic) ---
            if setup.reachable_face_ids and brep_graph is not None and graph_recognizer is not None:
                # Extract subgraph for reachable faces
                setup_subgraph = extract_subgraph(brep_graph, setup.reachable_face_ids)
                setup.brep_graph = setup_subgraph
                
                # Run graph-native diagnostics
                graph_diagnostics = graph_recognizer.analyze(setup_subgraph)
                setup.graph_feature_diagnostics = graph_diagnostics
                
                # Legacy field for backwards compatibility
                setup.brep_mfr_diagnostics = {fid: {'class': d.diagnostic_label, 'confidence': d.confidence} 
                                               for fid, d in graph_diagnostics.items()}
            # ----------------------------------------
            print(f"  -> {setup.summary()}", flush=True)
            # Print a few face IDs as sample
            if setup.reachable_face_ids:
                sample_ids = setup.reachable_face_ids[:5]
                print(f"     Sample Face IDs: {sample_ids} ...", flush=True)
            else:
                 print("     No reachable faces found.", flush=True)
            
            generated_setups.append(setup)
            
        print("=======================================================================", flush=True)

        # DeepMill Diagnostic Comparison
        print("\n=== DeepMill Diagnostic Comparison ===", flush=True)
        
        diag_means = []
        diag_maxs = []
        diag_mins = []
        
        for s in generated_setups:
            d = s.deepmill_diagnostics
            if d is not None:
                # Duck typing for tensor stats
                if hasattr(d, 'mean'): diag_means.append(float(d.mean()))
                if hasattr(d, 'max'): diag_maxs.append(float(d.max()))
                if hasattr(d, 'min'): diag_mins.append(float(d.min()))
        
        if diag_means:
            print(f"Cross-Setup Tensor Statistics (Neutral Observation):", flush=True)
            print(f"  - Total Setups with Diagnostics: {len(diag_means)}", flush=True)
            print(f"  - Signal Mean Variation: {min(diag_means):.4f} to {max(diag_means):.4f} (Delta: {max(diag_means)-min(diag_means):.4f})", flush=True)
            print(f"  - Signal Max Variation:  {min(diag_maxs):.4f} to {max(diag_maxs):.4f}", flush=True)
            print(f"  - Signal Min Variation:  {min(diag_mins):.4f} to {max(diag_mins):.4f}", flush=True)
        else:
            print("No diagnostic data available for comparison.", flush=True)
            
        print("======================================", flush=True)

        # Diagnostic Correlation (Non-Decision)
        print("\n=== Diagnostic Correlation (Non-Decision) ===", flush=True)
        if diag_means:
            print("Descriptive relationship between Geometric Area and DeepMill Signal Mean:", flush=True)
            
            # Pairs of (Area, Signal Mean)
            # Re-iterate setups to guarantee alignment
            pairs = []
            for s in generated_setups:
                 d = s.deepmill_diagnostics
                 if d is not None and hasattr(d, 'mean'):
                     pairs.append( (s.geometric_reachable_surface_area, float(d.mean())) )
            
            if pairs:
                # Sort by Geometric Area
                pairs.sort(key=lambda x: x[0])
                
                # Check for trend (naive)
                # Compare first half (low area) vs second half (high area) mean signals
                mid = len(pairs) // 2
                low_area_vals = [p[1] for p in pairs[:mid]]
                high_area_vals = [p[1] for p in pairs[mid:]]
                
                if low_area_vals and high_area_vals:
                    avg_low = sum(low_area_vals)/len(low_area_vals)
                    avg_high = sum(high_area_vals)/len(high_area_vals)
                    
                    trend = "Higher" if avg_high > avg_low else "Lower"
                    print(f"  - Setups with larger geometric reachable area tend to have {trend} signal means.", flush=True)
                    print(f"    (Avg Signal for Low Area: {avg_low:.4f} vs High Area: {avg_high:.4f})", flush=True)
                else:
                    print("  - Not enough data points to describe a trend.", flush=True)
            else:
                print("  - No paired data available.", flush=True)
        else:
            print("No diagnostic data available for correlation.", flush=True)
        print("=============================================", flush=True)
        
        # Print Execution Audit
        print(engine.get_audit_report(), flush=True)

    except Exception:
        print("Detailed Error Traceback:", flush=True)
        traceback.print_exc()
        sys.exit(1)

    print("Exiting cleanly.", flush=True)

if __name__ == "__main__":
    main()
