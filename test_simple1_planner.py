"""
Test script for simple 1.step - Setup Planner Analysis
"""
import sys
sys.path.insert(0, 'BREP Graph Generator')

from analysis_engine.brep_graph import BRepGraphGenerator
from analysis_engine.deepmill_setup_planner import DeepMillSetupPlanner, CoverageConfig

# Test Part
STEP_FILE = "Step Files/simple 1.step"

def main():
    print("="*60)
    print("SETUP PLANNER - Simple 1.step")
    print("="*60)
    
    # Load graph
    print("\nLoading B-Rep Graph...")
    generator = BRepGraphGenerator(use_fast_path=True)
    graph = generator.generate(STEP_FILE)
    print(f"Graph: {graph.num_faces} faces, {graph.num_edges} edges")
    
    # Print face type breakdown
    print("\nFace Types:")
    type_names = {0: 'Plane', 1: 'Cylinder', 2: 'Cone', 3: 'Sphere', 4: 'Torus', 5: 'BSpline', 6: 'Bezier', 7: 'Other'}
    type_counts = {}
    for face_id, face_data in graph.faces.items():
        st = face_data.get('surface_type', 7)
        type_counts[st] = type_counts.get(st, 0) + 1
    for st, count in sorted(type_counts.items()):
        print(f"  {type_names.get(st, 'Unknown')}: {count}")
    
    # Create mock geometry context
    class MinimalGeoContext:
        pass
    geo_context = MinimalGeoContext()
    
    # Run planner with refined config (15° tolerances)
    print("\n" + "="*60)
    print("RUNNING SETUP PLANNER (Refined Configuration)")
    print("="*60)
    config = CoverageConfig(
        plane_head_on_tolerance_deg=15.0,
        plane_side_cut_tolerance_deg=15.0,
        cylinder_axis_tolerance_deg=10.0,
        cone_axis_tolerance_deg=15.0
    )
    print(f"Tolerances: head-on={config.plane_head_on_tolerance_deg}°, side-cut={config.plane_side_cut_tolerance_deg}°, cyl={config.cylinder_axis_tolerance_deg}°")
    print()
    
    planner = DeepMillSetupPlanner()
    plan = planner.plan(geo_context, graph)
    
    print()
    print(plan.summary())
    
    # Print detailed spindle positions
    print("\n" + "="*60)
    print("SPINDLE POSITIONS REQUIRED")
    print("="*60)
    for op in plan.operations:
        axis = op.orientation
        print(f"\nSetup {op.operation_number}: {op.orientation_name}")
        print(f"  Tool Axis: ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
        print(f"  Faces machined in this setup: {len(op.newly_covered_faces)}")
        if len(op.newly_covered_faces) <= 10:
            print(f"  Face IDs: {sorted(op.newly_covered_faces)}")
        else:
            sample = sorted(list(op.newly_covered_faces))[:10]
            print(f"  Face IDs (first 10): {sample} ...")
    
    print("\n" + "="*60)
    print(f"TOTAL SPINDLE POSITIONS REQUIRED: {plan.total_setups}")
    print("="*60)

if __name__ == "__main__":
    main()
