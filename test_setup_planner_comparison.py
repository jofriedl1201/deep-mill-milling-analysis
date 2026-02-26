"""
Test script for DeepMill Setup Planner - Baseline vs Refined Comparison

Runs the setup planner on the DOOR MOTOR MOUNT part with:
1. Strict baseline configuration (0° tolerances)
2. Refined configuration (15° tolerances, adjacent access)

Reports:
- Total setup count for each
- Orientations selected
- Coverage reasons breakdown
- Faces that moved to earlier operations
"""
import sys
sys.path.insert(0, 'BREP Graph Generator')

from analysis_engine.brep_graph import BRepGraphGenerator
from analysis_engine.deepmill_setup_planner import DeepMillSetupPlanner, CoverageConfig

# Test Part
STEP_FILE = "Step Files/Left side Traitener.step"

def run_with_config(config, config_name):
    """Run planner with given configuration"""
    print(f"\n{'='*60}")
    print(f"CONFIGURATION: {config_name}")
    print(f"{'='*60}")
    print(f"  plane_head_on_tolerance_deg = {config.plane_head_on_tolerance_deg}")
    print(f"  plane_side_cut_tolerance_deg = {config.plane_side_cut_tolerance_deg}")
    print(f"  cylinder_axis_tolerance_deg = {config.cylinder_axis_tolerance_deg}")
    print()
    
    planner = DeepMillSetupPlanner()
    
    # Compute coverages with this config
    coverages = []
    for orientation, name in planner.orientations:
        result = planner.engine.predict(geo_context, orientation, 6.0)
        covered_faces, coverage_reasons = planner._project_voxels_to_faces(
            result, graph, orientation, config
        )
        
        from analysis_engine.deepmill_setup_planner import OrientationCoverage
        coverage = OrientationCoverage(
            orientation=orientation,
            orientation_name=name,
            covered_face_ids=covered_faces,
            coverage_ratio=len(covered_faces) / max(1, len(all_faces)),
            coverage_reasons=coverage_reasons
        )
        coverages.append(coverage)
        
        # Count access types
        access_counts = {}
        for reason in coverage_reasons.values():
            if reason.is_covered:
                access_counts[reason.access_type] = access_counts.get(reason.access_type, 0) + 1
        
        access_str = ", ".join([f"{k}:{v}" for k, v in sorted(access_counts.items())])
        print(f"  {name}: {len(covered_faces)}/{len(all_faces)} faces [{access_str}]")
    
    # Compute minimum setups
    plan = planner.compute_minimum_setups(coverages, all_faces)
    
    print()
    print(plan.summary())
    
    return plan, coverages


def main():
    print("="*60)
    print("DEEPMILL SETUP PLANNER - BASELINE vs REFINED COMPARISON")
    print("="*60)
    print(f"Part: Left side Traitener")
    print(f"File: {STEP_FILE}")
    print()
    
    # Load graph (global for both runs)
    print("Loading B-Rep Graph...")
    generator = BRepGraphGenerator(use_fast_path=True)
    global graph, all_faces, geo_context
    graph = generator.generate(STEP_FILE)
    all_faces = set(graph.get_face_ids())
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
    
    # Configuration 1: STRICT BASELINE (0° tolerances)
    baseline_config = CoverageConfig(
        plane_head_on_tolerance_deg=0.0,
        plane_side_cut_tolerance_deg=0.0,
        cylinder_axis_tolerance_deg=0.0,
        cone_axis_tolerance_deg=0.0
    )
    baseline_plan, baseline_coverages = run_with_config(baseline_config, "STRICT BASELINE")
    
    # Configuration 2: REFINED (15° tolerances, adjacent access enabled)
    refined_config = CoverageConfig(
        plane_head_on_tolerance_deg=15.0,
        plane_side_cut_tolerance_deg=15.0,
        cylinder_axis_tolerance_deg=10.0,
        cone_axis_tolerance_deg=15.0
    )
    refined_plan, refined_coverages = run_with_config(refined_config, "REFINED")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Baseline Setups: {baseline_plan.total_setups}")
    print(f"Refined Setups:  {refined_plan.total_setups}")
    print(f"Reduction:       {baseline_plan.total_setups - refined_plan.total_setups} setup(s)")
    print()
    
    # Track which operation each face was covered in
    baseline_face_to_op = {}
    for op in baseline_plan.operations:
        for face_id in op.newly_covered_faces:
            baseline_face_to_op[face_id] = op.operation_number
    
    refined_face_to_op = {}
    for op in refined_plan.operations:
        for face_id in op.newly_covered_faces:
            refined_face_to_op[face_id] = op.operation_number
    
    # Find faces that moved to earlier operations
    moved_earlier = []
    for face_id in all_faces:
        baseline_op = baseline_face_to_op.get(face_id)
        refined_op = refined_face_to_op.get(face_id)
        if baseline_op is not None and refined_op is not None:
            if refined_op < baseline_op:
                moved_earlier.append((face_id, baseline_op, refined_op))
    
    if moved_earlier:
        print(f"Faces moved to EARLIER operations: {len(moved_earlier)}")
        for face_id, baseline_op, refined_op in sorted(moved_earlier, key=lambda x: (x[2], x[0])):
            print(f"  Face {face_id}: Op{baseline_op} -> Op{refined_op}")
    else:
        print("No faces moved to earlier operations.")
    
    print()
    print("="*60)
    print("DETAILED COVERAGE BY OPERATION (REFINED)")
    print("="*60)
    
    for op in refined_plan.operations:
        print(f"\nOperation {op.operation_number}: {op.orientation_name}")
        
        # Get coverage reasons for this orientation
        cov = next((c for c in refined_coverages if c.orientation_name == op.orientation_name), None)
        if cov:
            access_breakdown = {}
            for face_id in op.newly_covered_faces:
                reason = cov.coverage_reasons.get(face_id)
                if reason and reason.is_covered:
                    access_breakdown[reason.access_type] = access_breakdown.get(reason.access_type, 0) + 1
            
            print(f"  New faces covered: {len(op.newly_covered_faces)}")
            print(f"  Access types:")
            for access_type, count in sorted(access_breakdown.items()):
                print(f"    {access_type}: {count} faces")

if __name__ == "__main__":
    main()
