"""
Manufacturing Analysis Module

Answers core manufacturing questions:
- Minimum orientations needed
- Features per orientation  
- Tool constraints per setup
- Surface finishing breakdown
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import math


@dataclass
class FeatureInfo:
    """Extracted feature with manufacturing-relevant properties."""
    face_id: int
    surface_type: int  # 0=Plane, 1=Cylinder, etc.
    area: float
    access_direction: Tuple[float, float, float]  # Normal or axis direction
    
    # Dimensional data (from geom_params)
    diameter: Optional[float] = None  # For cylinders
    radius: Optional[float] = None
    depth: Optional[float] = None  # Computed from geometry
    
    # Classification
    feature_type: str = "unknown"  # hole, pocket, boss, wall, floor, fillet, etc.
    requires_ball_endmill: bool = False


@dataclass  
class SetupPlan:
    """Manufacturing plan for a single setup/orientation."""
    orientation: Tuple[float, float, float]
    is_principal: bool  # True if aligned with X, Y, or Z axis
    
    # Assigned features
    feature_ids: List[int] = field(default_factory=list)
    features: List[FeatureInfo] = field(default_factory=list)
    
    # Constraints
    max_endmill_diameter: float = float('inf')
    min_tool_reach: float = 0.0
    
    # Areas
    finish_area_total: float = 0.0
    finish_area_bottom: float = 0.0  # Horizontal (floor) cuts
    finish_area_side: float = 0.0    # Vertical (wall) cuts  
    finish_area_ball: float = 0.0    # Sculptured/fillet cuts
    
    # Volume (to be computed)
    material_removal_mm3: float = 0.0


@dataclass
class ManufacturingPlan:
    """Complete manufacturing plan for a part."""
    setups: List[SetupPlan] = field(default_factory=list)
    total_orientations: int = 0
    requires_5axis: bool = False
    smallest_feature_diameter: float = float('inf')
    deepest_feature: float = 0.0
    
    def summary(self) -> str:
        lines = [
            f"=== Manufacturing Plan ===",
            f"Total Setups: {self.total_orientations}",
            f"5-Axis Required: {self.requires_5axis}",
            f"Smallest Feature: Ø{self.smallest_feature_diameter:.2f} mm",
            f"Deepest Feature: {self.deepest_feature:.2f} mm",
            ""
        ]
        for i, s in enumerate(self.setups):
            axis_str = f"({s.orientation[0]:.2f}, {s.orientation[1]:.2f}, {s.orientation[2]:.2f})"
            aligned = "XYZ-aligned" if s.is_principal else "TILTED"
            lines.append(f"Setup {i+1}: {axis_str} [{aligned}]")
            lines.append(f"  Features: {len(s.features)}")
            lines.append(f"  Max Endmill: Ø{s.max_endmill_diameter:.2f} mm" if s.max_endmill_diameter < float('inf') else "  Max Endmill: Unconstrained")
            lines.append(f"  Finish Area: {s.finish_area_total:.1f} mm² (Bottom: {s.finish_area_bottom:.1f}, Side: {s.finish_area_side:.1f}, Ball: {s.finish_area_ball:.1f})")
        return "\n".join(lines)


class ManufacturingAnalyzer:
    """
    Analyzes B-Rep graph to produce manufacturing plan.
    
    Usage:
        analyzer = ManufacturingAnalyzer()
        plan = analyzer.analyze(brep_graph)
        print(plan.summary())
    """
    
    SURFACE_TYPE_NAMES = {
        0: "Plane", 1: "Cylinder", 2: "Cone", 3: "Sphere",
        4: "Torus", 5: "Bezier", 6: "BSpline", 7: "Other"
    }
    
    def __init__(self):
        self.tolerance = 0.01  # For direction comparison
    
    def analyze(self, graph) -> ManufacturingPlan:
        """
        Analyze a BRepGraph and produce a manufacturing plan.
        
        Args:
            graph: BRepGraph with geom_params populated
            
        Returns:
            ManufacturingPlan with setups, constraints, and areas
        """
        # Step 1: Extract features from graph
        features = self._extract_features(graph)
        
        # Step 2: Determine required orientations
        orientations = self._compute_orientations(features)
        
        # Step 3: Assign features to setups
        setups = self._assign_features_to_setups(features, orientations)
        
        # Step 4: Compute constraints per setup
        for setup in setups:
            self._compute_setup_constraints(setup)
        
        # Step 5: Build manufacturing plan
        plan = ManufacturingPlan(
            setups=setups,
            total_orientations=len(setups),
            requires_5axis=any(not s.is_principal for s in setups),
            smallest_feature_diameter=min((f.diameter for f in features if f.diameter), default=float('inf')),
            deepest_feature=max((f.depth for f in features if f.depth), default=0.0)
        )
        
        return plan
    
    def _extract_features(self, graph) -> List[FeatureInfo]:
        """Extract manufacturing features from graph faces."""
        features = []
        
        for face_id, face_data in graph.faces.items():
            surface_type = face_data.get("surface_type", 7)
            area = face_data.get("area", 0.0)
            geom_params = face_data.get("geom_params", {})
            
            # Determine access direction based on surface type
            if surface_type == 0:  # Plane
                access_dir = tuple(geom_params.get("normal", [0, 0, 1]))
            elif surface_type == 1:  # Cylinder
                access_dir = tuple(geom_params.get("axis_direction", [0, 0, 1]))
            elif surface_type == 2:  # Cone
                access_dir = tuple(geom_params.get("axis_direction", [0, 0, 1]))
            elif surface_type == 3:  # Sphere
                # Sphere accessible from any direction - use Z as default
                access_dir = (0.0, 0.0, 1.0)
            elif surface_type == 4:  # Torus (fillet)
                access_dir = tuple(geom_params.get("axis_direction", [0, 0, 1]))
            else:
                access_dir = (0.0, 0.0, 1.0)  # Default
            
            feature = FeatureInfo(
                face_id=face_id,
                surface_type=surface_type,
                area=area,
                access_direction=access_dir,
                diameter=geom_params.get("diameter"),
                radius=geom_params.get("radius")
            )
            
            # Classify feature type
            feature.feature_type = self._classify_feature(surface_type, geom_params)
            feature.requires_ball_endmill = surface_type in [3, 4, 5, 6]  # Sphere, Torus, Bezier, BSpline
            
            features.append(feature)
        
        return features
    
    def _classify_feature(self, surface_type: int, geom_params: dict) -> str:
        """Classify a face into manufacturing feature type."""
        if surface_type == 0:  # Plane
            normal = geom_params.get("normal", [0, 0, 1])
            # Check if horizontal (floor) or vertical (wall)
            if abs(normal[2]) > 0.9:
                return "floor"
            elif abs(normal[2]) < 0.1:
                return "wall"
            else:
                return "angled_face"
        elif surface_type == 1:  # Cylinder
            return "hole_wall"
        elif surface_type == 2:  # Cone
            return "chamfer"
        elif surface_type == 3:  # Sphere
            return "ball_feature"
        elif surface_type == 4:  # Torus
            return "fillet"
        elif surface_type in [5, 6]:  # Bezier/BSpline
            return "sculptured"
        else:
            return "unknown"
    
    def _compute_orientations(self, features: List[FeatureInfo]) -> List[Tuple[Tuple[float, float, float], bool]]:
        """
        Compute minimum set of orientations needed.
        
        Returns list of (direction, is_principal) tuples.
        """
        # Collect unique access directions
        directions: Set[Tuple[float, float, float]] = set()
        
        for f in features:
            # Normalize direction
            d = f.access_direction
            mag = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
            if mag > 0.001:
                normalized = (d[0]/mag, d[1]/mag, d[2]/mag)
                # Round to avoid floating point duplicates
                rounded = (round(normalized[0], 2), round(normalized[1], 2), round(normalized[2], 2))
                directions.add(rounded)
        
        # Determine if each is principal (XYZ-aligned)
        principal_axes = [
            (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
        ]
        
        result = []
        for d in directions:
            is_principal = any(self._directions_equal(d, p) for p in principal_axes)
            result.append((d, is_principal))
        
        return result
    
    def _directions_equal(self, d1: Tuple[float, float, float], d2: Tuple[float, float, float]) -> bool:
        """Check if two directions are equal within tolerance."""
        return all(abs(a - b) < self.tolerance for a, b in zip(d1, d2))
    
    def _assign_features_to_setups(self, features: List[FeatureInfo], 
                                    orientations: List[Tuple[Tuple[float, float, float], bool]]) -> List[SetupPlan]:
        """Assign each feature to appropriate setup."""
        setups = []
        
        for orientation, is_principal in orientations:
            setup = SetupPlan(
                orientation=orientation,
                is_principal=is_principal
            )
            
            # Find features accessible from this orientation
            for f in features:
                if self._directions_equal(f.access_direction, orientation) or \
                   self._directions_equal(f.access_direction, (-orientation[0], -orientation[1], -orientation[2])):
                    setup.feature_ids.append(f.face_id)
                    setup.features.append(f)
            
            if setup.features:  # Only add non-empty setups
                setups.append(setup)
        
        return setups
    
    def _compute_setup_constraints(self, setup: SetupPlan):
        """Compute tool constraints and surface areas for a setup."""
        # Max endmill diameter (constrained by smallest hole)
        hole_diameters = [f.diameter for f in setup.features if f.diameter and f.feature_type == "hole_wall"]
        if hole_diameters:
            setup.max_endmill_diameter = min(hole_diameters)
        
        # Surface area breakdown
        for f in setup.features:
            setup.finish_area_total += f.area
            
            if f.feature_type in ["floor"]:
                setup.finish_area_bottom += f.area
            elif f.feature_type in ["wall", "hole_wall"]:
                setup.finish_area_side += f.area
            elif f.requires_ball_endmill:
                setup.finish_area_ball += f.area
            else:
                # Default to side cut for angled/unknown
                setup.finish_area_side += f.area


def analyze_part(step_path: str) -> ManufacturingPlan:
    """
    Convenience function to analyze a STEP file.
    
    Args:
        step_path: Path to STEP file
        
    Returns:
        ManufacturingPlan with complete analysis
    """
    import sys
    sys.path.insert(0, 'BREP Graph Generator')
    from analysis_engine.brep_graph import BRepGraphGenerator
    
    generator = BRepGraphGenerator(use_fast_path=True)
    graph = generator.generate(step_path)
    
    analyzer = ManufacturingAnalyzer()
    return analyzer.analyze(graph)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        plan = analyze_part(sys.argv[1])
        print(plan.summary())
    else:
        print("Usage: python manufacturing_analyzer.py <step_file>")
