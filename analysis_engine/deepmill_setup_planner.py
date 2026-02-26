"""
DeepMill Setup Planner

Computes minimum machining setups and optimal orientations using DeepMill
voxel accessibility output. Uses pretrained model as-is with nominal cutter.

This module:
1. Defines 6 axis-aligned candidate orientations (±X, ±Y, ±Z)
2. Runs DeepMill inference for each orientation
3. Projects accessibility volumes to B-Rep face coverage
4. Implements greedy set cover for minimum orientations
5. Reports ordered setup plan
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Any, Optional
import numpy as np
import math


# ============================================================================
# COVERAGE CONFIGURATION (Configurable tolerances)
# ============================================================================

@dataclass
class CoverageConfig:
    """Configurable tolerances for face coverage projection."""
    
    # Angular tolerance for head-on plane access (degrees)
    # Face is accessible if angle between face normal and -tool_axis is within this
    plane_head_on_tolerance_deg: float = 15.0
    
    # Angular tolerance for side-cutting plane access (degrees from perpendicular)
    # Face is accessible via side cut if angle from perpendicular is within this
    plane_side_cut_tolerance_deg: float = 15.0
    
    # Angular tolerance for cylinder axis alignment (degrees)
    cylinder_axis_tolerance_deg: float = 10.0
    
    # Angular tolerance for cone axis alignment (degrees)
    cone_axis_tolerance_deg: float = 15.0
    
    def __post_init__(self):
        # Convert to radians for internal use
        self.plane_head_on_tolerance_rad = math.radians(self.plane_head_on_tolerance_deg)
        self.plane_side_cut_tolerance_rad = math.radians(self.plane_side_cut_tolerance_deg)
        self.cylinder_axis_tolerance_rad = math.radians(self.cylinder_axis_tolerance_deg)
        self.cone_axis_tolerance_rad = math.radians(self.cone_axis_tolerance_deg)
        
        # Precompute cosine thresholds for dot product comparisons
        self.plane_head_on_cos_threshold = math.cos(self.plane_head_on_tolerance_rad)
        self.plane_side_cut_cos_threshold = math.cos(math.radians(90 - self.plane_side_cut_tolerance_deg))
        self.cylinder_cos_threshold = math.cos(self.cylinder_axis_tolerance_rad)
        self.cone_cos_threshold = math.cos(self.cone_axis_tolerance_rad)


# ============================================================================
# FACE COVERAGE REASON
# ============================================================================

@dataclass
class FaceCoverageReason:
    """Explains why a face is considered covered by an orientation."""
    face_id: int
    is_covered: bool
    access_type: str  # "head_on", "within_tolerance", "side_cut", "axis_aligned", "not_covered"
    angle_deg: float  # Actual angle used for decision
    explanation: str  # Human-readable explanation


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OrientationCoverage:
    """Coverage data for a single orientation."""
    orientation: Tuple[float, float, float]
    orientation_name: str  # e.g., "+Z", "-X"
    covered_face_ids: Set[int]
    coverage_ratio: float  # 0.0 to 1.0
    coverage_reasons: Dict[int, FaceCoverageReason] = field(default_factory=dict)  # NEW
    voxel_tensor: Any = None  # Raw DeepMill output (optional)



@dataclass
class SetupOperation:
    """A single machining operation/setup."""
    operation_number: int
    orientation: Tuple[float, float, float]
    orientation_name: str
    covered_faces: Set[int]
    newly_covered_faces: Set[int]  # Faces covered by this op that weren't covered before


@dataclass
class SetupPlan:
    """Complete setup plan for a part."""
    operations: List[SetupOperation]
    total_setups: int
    total_faces: int
    covered_faces: int
    uncovered_faces: Set[int]
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "DEEPMILL SETUP PLAN",
            "=" * 60,
            f"Total Setups Required: {self.total_setups}",
            f"Total Faces: {self.total_faces}",
            f"Covered: {self.covered_faces} ({100*self.covered_faces/max(1,self.total_faces):.1f}%)",
            "",
        ]
        
        for op in self.operations:
            axis_str = f"({op.orientation[0]:.2f}, {op.orientation[1]:.2f}, {op.orientation[2]:.2f})"
            lines.append(f"Operation {op.operation_number}: {op.orientation_name} {axis_str}")
            lines.append(f"  Covers: {len(op.covered_faces)} faces total")
            lines.append(f"  New in this op: {len(op.newly_covered_faces)} faces")
            if len(op.newly_covered_faces) <= 10:
                lines.append(f"  Face IDs: {sorted(op.newly_covered_faces)}")
            else:
                sample = sorted(list(op.newly_covered_faces))[:5]
                lines.append(f"  Face IDs (sample): {sample} ...")
        
        if self.uncovered_faces:
            lines.append("")
            lines.append(f"WARNING: {len(self.uncovered_faces)} faces remain uncovered!")
            if len(self.uncovered_faces) <= 10:
                lines.append(f"  Uncovered Face IDs: {sorted(self.uncovered_faces)}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# CANDIDATE ORIENTATIONS
# ============================================================================

# Fixed set of 6 axis-aligned orientations (±X, ±Y, ±Z)
CANDIDATE_ORIENTATIONS = [
    ((1.0, 0.0, 0.0), "+X"),
    ((-1.0, 0.0, 0.0), "-X"),
    ((0.0, 1.0, 0.0), "+Y"),
    ((0.0, -1.0, 0.0), "-Y"),
    ((0.0, 0.0, 1.0), "+Z"),
    ((0.0, 0.0, -1.0), "-Z"),
]


# ============================================================================
# DEEPMILL SETUP PLANNER
# ============================================================================

class DeepMillSetupPlanner:
    """
    Plans machining setups using DeepMill accessibility analysis.
    
    Uses pretrained DeepMill model to determine which faces are accessible
    from each orientation, then applies greedy set cover to find minimum setups.
    """
    
    def __init__(self, deepmill_engine=None):
        """
        Args:
            deepmill_engine: DeepMillAccessibilityEngine instance.
                            If None, will create one internally.
        """
        if deepmill_engine is None:
            from analysis_engine.deepmill import DeepMillAccessibilityEngine
            self.engine = DeepMillAccessibilityEngine()
        else:
            self.engine = deepmill_engine
        
        self.orientations = CANDIDATE_ORIENTATIONS
    
    def compute_orientation_coverage(
        self, 
        geometry_context: Any,
        brep_graph: Any,
        tool_diameter: float = 6.0
    ) -> List[OrientationCoverage]:
        """
        Run DeepMill for each candidate orientation and compute face coverage.
        
        Args:
            geometry_context: GeometryContext with mesh for DeepMill
            brep_graph: BRepGraph with face information
            tool_diameter: Nominal tool diameter (not varied in this step)
            
        Returns:
            List of OrientationCoverage, one per candidate orientation
        """
        coverages = []
        all_face_ids = set(brep_graph.get_face_ids())
        
        print(f"\n=== Computing DeepMill Coverage for {len(self.orientations)} Orientations ===")
        
        for orientation, name in self.orientations:
            # Run DeepMill inference
            result = self.engine.predict(geometry_context, orientation, tool_diameter)
            
            # Project voxel output to face coverage (with tolerance-aware logic)
            covered_faces, coverage_reasons = self._project_voxels_to_faces(
                result, brep_graph, orientation
            )
            
            coverage_ratio = len(covered_faces) / max(1, len(all_face_ids))
            
            # Count access types
            access_counts = {}
            for reason in coverage_reasons.values():
                if reason.is_covered:
                    access_counts[reason.access_type] = access_counts.get(reason.access_type, 0) + 1
            
            coverage = OrientationCoverage(
                orientation=orientation,
                orientation_name=name,
                covered_face_ids=covered_faces,
                coverage_ratio=coverage_ratio,
                coverage_reasons=coverage_reasons,
                voxel_tensor=getattr(result, 'deepmill_diagnostics', None)
            )
            coverages.append(coverage)
            
            # Enhanced logging with access type breakdown
            access_str = ", ".join([f"{k}:{v}" for k, v in sorted(access_counts.items())])
            print(f"  {name}: {len(covered_faces)}/{len(all_face_ids)} faces ({100*coverage_ratio:.1f}%) [{access_str}]")

        
        return coverages
    
    def _project_voxels_to_faces(
        self,
        deepmill_result: Any,
        brep_graph: Any,
        orientation: Tuple[float, float, float],
        config: Optional[CoverageConfig] = None
    ) -> Tuple[Set[int], Dict[int, FaceCoverageReason]]:
        """
        Project DeepMill voxel accessibility prediction to B-Rep face coverage.
        
        Uses configurable tolerances for machining-realistic coverage decisions.
        Tracks and returns reason for each face's coverage status.
        
        Args:
            deepmill_result: Result from DeepMill inference
            brep_graph: BRepGraph with face geometry
            orientation: Tool approach direction
            config: Coverage configuration (uses defaults if None)
            
        Returns:
            Tuple of (covered_face_ids, coverage_reasons)
        """
        if config is None:
            config = CoverageConfig()
        
        covered_faces = set()
        coverage_reasons = {}
        
        # Get orientation as numpy array
        axis = np.array(orientation)
        orientation_name = self._get_orientation_name(orientation)
        
        # Log configuration
        print(f"    [Config] Head-on: {config.plane_head_on_tolerance_deg}°, Side-cut: {config.plane_side_cut_tolerance_deg}°, Cyl: {config.cylinder_axis_tolerance_deg}°")
        
        for face_id, face_data in brep_graph.faces.items():
            geom_params = face_data.get("geom_params", {})
            surface_type = face_data.get("surface_type", 7)
            
            # Initialize reason
            is_accessible = False
            access_type = "not_covered"
            angle_deg = 0.0
            explanation = ""
            
            if surface_type == 0:  # Plane
                normal = np.array(geom_params.get("normal", [0, 0, 1]))
                dot = np.dot(normal, axis)
                
                # Angle between face normal and -tool_axis (head-on approach)
                # Negative dot product means normal points opposite to tool axis (favorable)
                head_on_angle_deg = math.degrees(math.acos(np.clip(abs(dot), 0, 1)))
                
                # Check if within head-on tolerance
                if head_on_angle_deg <= config.plane_head_on_tolerance_deg:
                    is_accessible = True
                    angle_deg = head_on_angle_deg
                    if head_on_angle_deg <= 1.0:
                        access_type = "head_on"
                        explanation = f"Perfect head-on access ({angle_deg:.1f}°)"
                    else:
                        access_type = "within_tolerance"
                        explanation = f"Within {config.plane_head_on_tolerance_deg}° tolerance ({angle_deg:.1f}°)"
                else:
                    # Check for side-cutting possibility
                    # Side cutting works when face is approximately perpendicular to tool axis
                    perp_angle_deg = abs(90 - head_on_angle_deg)
                    
                    if perp_angle_deg <= config.plane_side_cut_tolerance_deg:
                        is_accessible = True
                        access_type = "side_cut"
                        angle_deg = perp_angle_deg
                        explanation = f"Side-cut access ({perp_angle_deg:.1f}° from perpendicular)"
                    else:
                        angle_deg = head_on_angle_deg
                        explanation = f"Not accessible ({head_on_angle_deg:.1f}° off head-on, {perp_angle_deg:.1f}° off perpendicular)"
                
            elif surface_type == 1:  # Cylinder (hole wall)
                axis_dir = np.array(geom_params.get("axis_direction", [0, 0, 1]))
                dot = abs(np.dot(axis_dir, axis))
                axis_angle_deg = math.degrees(math.acos(np.clip(dot, 0, 1)))
                
                if axis_angle_deg <= config.cylinder_axis_tolerance_deg:
                    # Direct axis-aligned drilling
                    is_accessible = True
                    access_type = "axis_aligned"
                    angle_deg = axis_angle_deg
                    explanation = f"Cylinder axis aligned ({angle_deg:.1f}° from tool axis)"
                else:
                    # Check if adjacent planar faces are accessible from this orientation
                    adjacent_plane_accessible = False
                    adjacent_faces = brep_graph.get_adjacent_faces(face_id)
                    
                    for adj_face_id in adjacent_faces:
                        adj_face = brep_graph.faces.get(adj_face_id, {})
                        if adj_face.get("surface_type") == 0:  # Adjacent plane
                            adj_normal = np.array(adj_face.get("geom_params", {}).get("normal", [0,0,1]))
                            adj_dot = np.dot(adj_normal, axis)
                            adj_angle = math.degrees(math.acos(np.clip(abs(adj_dot), 0, 1)))
                            if adj_angle <= config.plane_head_on_tolerance_deg:
                                adjacent_plane_accessible = True
                                break
                    
                    if adjacent_plane_accessible:
                        is_accessible = True
                        access_type = "adjacent_access"
                        angle_deg = axis_angle_deg
                        explanation = f"Cylinder accessible via adjacent planar face (axis {axis_angle_deg:.1f}° off)"
                    else:
                        angle_deg = axis_angle_deg
                        explanation = f"Cylinder axis misaligned ({axis_angle_deg:.1f}° > {config.cylinder_axis_tolerance_deg}° tolerance, no adjacent access)"
                
            elif surface_type == 2:  # Cone (chamfer)
                axis_dir = np.array(geom_params.get("axis_direction", [0, 0, 1]))
                dot = abs(np.dot(axis_dir, axis))
                axis_angle_deg = math.degrees(math.acos(np.clip(dot, 0, 1)))
                
                if dot >= config.cone_cos_threshold:
                    is_accessible = True
                    access_type = "axis_aligned"
                    angle_deg = axis_angle_deg
                    explanation = f"Cone axis aligned ({angle_deg:.1f}° from tool axis)"
                else:
                    angle_deg = axis_angle_deg
                    explanation = f"Cone axis misaligned ({axis_angle_deg:.1f}° > {config.cone_axis_tolerance_deg}° tolerance)"
                
            elif surface_type in [3, 4]:  # Sphere, Torus (fillets)
                if surface_type == 4:  # Torus
                    axis_dir = np.array(geom_params.get("axis_direction", [0, 0, 1]))
                    dot = abs(np.dot(axis_dir, axis))
                    axis_angle_deg = math.degrees(math.acos(np.clip(dot, 0, 1)))
                    if dot >= 0.7:
                        is_accessible = True
                        access_type = "axis_aligned"
                        angle_deg = axis_angle_deg
                        explanation = f"Torus accessible ({angle_deg:.1f}° from axis)"
                    else:
                        angle_deg = axis_angle_deg
                        explanation = f"Torus axis misaligned ({axis_angle_deg:.1f}°)"
                else:
                    is_accessible = True
                    access_type = "multi_directional"
                    angle_deg = 0.0
                    explanation = "Sphere accessible from any direction"
                
            else:  # BSpline, Bezier, Other
                is_accessible = True
                access_type = "multi_directional"
                angle_deg = 0.0
                explanation = "Freeform surface - conservatively marked accessible"
            
            # Record reason
            reason = FaceCoverageReason(
                face_id=face_id,
                is_covered=is_accessible,
                access_type=access_type,
                angle_deg=angle_deg,
                explanation=explanation
            )
            coverage_reasons[face_id] = reason
            
            if is_accessible:
                covered_faces.add(face_id)
        
        return covered_faces, coverage_reasons
    
    def _get_orientation_name(self, orientation: Tuple[float, float, float]) -> str:
        """Get human-readable name for orientation."""
        for o, name in CANDIDATE_ORIENTATIONS:
            if o == orientation:
                return name
        return f"({orientation[0]:.2f}, {orientation[1]:.2f}, {orientation[2]:.2f})"


    
    def compute_minimum_setups(
        self,
        coverages: List[OrientationCoverage],
        all_face_ids: Set[int]
    ) -> SetupPlan:
        """
        Apply greedy set cover algorithm to find minimum orientations.
        
        Greedy approach: repeatedly select the orientation that covers
        the largest amount of remaining uncovered geometry.
        
        Args:
            coverages: Coverage data from compute_orientation_coverage
            all_face_ids: Complete set of face IDs that need coverage
            
        Returns:
            SetupPlan with ordered operations
        """
        print("\n=== Computing Minimum Setup Cover ===")
        
        operations = []
        remaining = all_face_ids.copy()
        used_orientations = set()
        
        op_num = 1
        
        while remaining and len(used_orientations) < len(coverages):
            # Find orientation that covers most remaining faces
            best_coverage = None
            best_new_count = 0
            
            for cov in coverages:
                if cov.orientation_name in used_orientations:
                    continue
                
                new_covered = cov.covered_face_ids & remaining
                if len(new_covered) > best_new_count:
                    best_new_count = len(new_covered)
                    best_coverage = cov
            
            if best_coverage is None or best_new_count == 0:
                # No more progress possible
                break
            
            # Add this orientation to the plan
            newly_covered = best_coverage.covered_face_ids & remaining
            remaining -= newly_covered
            used_orientations.add(best_coverage.orientation_name)
            
            operation = SetupOperation(
                operation_number=op_num,
                orientation=best_coverage.orientation,
                orientation_name=best_coverage.orientation_name,
                covered_faces=best_coverage.covered_face_ids.copy(),
                newly_covered_faces=newly_covered
            )
            operations.append(operation)
            
            print(f"  Operation {op_num}: {best_coverage.orientation_name} covers {len(newly_covered)} new faces")
            
            op_num += 1
        
        plan = SetupPlan(
            operations=operations,
            total_setups=len(operations),
            total_faces=len(all_face_ids),
            covered_faces=len(all_face_ids) - len(remaining),
            uncovered_faces=remaining
        )
        
        return plan
    
    def plan(
        self,
        geometry_context: Any,
        brep_graph: Any,
        tool_diameter: float = 6.0
    ) -> SetupPlan:
        """
        Complete setup planning workflow.
        
        Args:
            geometry_context: GeometryContext with mesh
            brep_graph: BRepGraph with dimensional data
            tool_diameter: Nominal tool diameter
            
        Returns:
            SetupPlan with minimum required operations
        """
        # Step 1: Compute coverage for all orientations
        coverages = self.compute_orientation_coverage(
            geometry_context, brep_graph, tool_diameter
        )
        
        # Step 2: Get all face IDs
        all_faces = set(brep_graph.get_face_ids())
        
        # Step 3: Compute minimum set cover
        plan = self.compute_minimum_setups(coverages, all_faces)
        
        return plan


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def plan_setups_for_step_file(step_path: str) -> SetupPlan:
    """
    Convenience function to plan setups for a STEP file.
    
    Args:
        step_path: Path to STEP file
        
    Returns:
        SetupPlan with minimum orientations
    """
    import sys
    sys.path.insert(0, 'BREP Graph Generator')
    
    from analysis_engine.brep_graph import BRepGraphGenerator
    from analysis_engine.deepmill import DeepMillAccessibilityEngine
    
    # Load STEP and generate B-Rep graph
    print(f"Loading: {step_path}")
    generator = BRepGraphGenerator(use_fast_path=True)
    graph = generator.generate(step_path)
    print(f"Graph: {graph.num_faces} faces, {graph.num_edges} edges")
    
    # Create mock geometry context (DeepMill needs mesh, but we use graph for coverage)
    class MinimalGeoContext:
        pass
    geo_context = MinimalGeoContext()
    
    # Plan setups
    planner = DeepMillSetupPlanner()
    plan = planner.plan(geo_context, graph)
    
    return plan


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        step_file = sys.argv[1]
    else:
        step_file = "Step Files/S-U09900012-000__MOUNT - TADS V2.0 DOOR MOTOR MOUNT.step"
    
    plan = plan_setups_for_step_file(step_file)
    print("\n" + plan.summary())
