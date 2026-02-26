"""
Test script for DeepMill Setup Planner

Runs the setup planner on the DOOR MOTOR MOUNT part and outputs:
1. Total setup count
2. Orientations selected
3. Coverage reasons for each face
"""
import sys
sys.path.insert(0, 'BREP Graph Generator')

from analysis_engine.brep_graph import BRepGraphGenerator
from analysis_engine.deepmill_setup_planner import DeepMillSetupPlanner, CoverageConfig

# Test Part
STEP_FILE = "Step Files/S-U09900012-000__MOUNT - TADS V2.0 DOOR MOTOR MOUNT.step"

def main():
    print("=" * 60)
    print("DEEPMILL SETUP PLANNER - BASELINE TEST")
    print("=" * 60)
    print(f"Part: DOOR MOTOR MOUNT")
    print(f"File: {STEP_FILE}")
    print()
    
    # Load graph
    print("Loading B-Rep Graph...")
    generator = BRepGraphGenerator(use_fast_path=True)
    graph = generator.generate(STEP_FILE)
    print(f"Graph: {graph.num_faces} faces, {graph.num_edges} edges")
    print()
    
    # Print face type breakdown
    print("Face Types:")
    type_names = {0: 'Plane', 1: 'Cylinder', 2: 'Cone', 3: 'Sphere', 4: 'Torus', 5: 'BSpline', 6: 'Bezier', 7: 'Other'}
    type_counts = {}
    for face_id, face_data in graph.faces.items():
        st = face_data.get('surface_type', 7)
        type_counts[st] = type_counts.get(st, 0) + 1
    for st, count in sorted(type_counts.items()):
        print(f"  {type_names.get(st, 'Unknown')}: {count}")
    print()
    
    # Create mock geometry context
    class MinimalGeoContext:
        pass
    geo_context = MinimalGeoContext()
    
    # Run planner with default config
    print("Running Setup Planner with DEFAULT config:")
    config = CoverageConfig()
    print(f"  plane_head_on_tolerance_deg = {config.plane_head_on_tolerance_deg}")
    print(f"  plane_side_cut_tolerance_deg = {config.plane_side_cut_tolerance_deg}")
    print(f"  cylinder_axis_tolerance_deg = {config.cylinder_axis_tolerance_deg}")
    print(f"  cone_axis_tolerance_deg = {config.cone_axis_tolerance_deg}")
    print()
    
    planner = DeepMillSetupPlanner()
    plan = planner.plan(geo_context, graph)
    
    print()
    print(plan.summary())
    
    # Print coverage breakdown for each operation
    print()
    print("=" * 60)
    print("COVERAGE REASONS BY OPERATION")
    print("=" * 60)
    
    # Get all coverages with reasons
    coverages = planner.compute_orientation_coverage(geo_context, graph)
    
    for cov in coverages:
        if cov.orientation_name in [op.orientation_name for op in plan.operations]:
            print(f"\nOrientation {cov.orientation_name}:")
            access_types = {}
            for reason in cov.coverage_reasons.values():
                if reason.is_covered:
                    at = reason.access_type
                    access_types[at] = access_types.get(at, 0) + 1
            for at, count in sorted(access_types.items()):
                print(f"  {at}: {count} faces")

if __name__ == "__main__":
    main()
