from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional, TYPE_CHECKING
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

if TYPE_CHECKING:
    from .brep_graph import BRepGraph

@dataclass
class SetupContext:
    """
    Represents a single machining setup configuration.
    Semantics: PURE GEOMETRY ONLY. No costing/time attributes.
    """
    orientation_vector: Tuple[float, float, float]
    tool_diameter: float
    # renamed to explicitly indicate this is a reachability set, not a process plan
    reachable_face_ids: List[int] = field(default_factory=list)
    # renamed to explicitly indicate this is a geometric property
    geometric_reachable_surface_area: float = 0.0
    # explicitly labeled as a reachability volume approximation
    # breakdown of reachable area by orientation
    geometric_aligned_surface_area: float = 0.0
    geometric_perpendicular_surface_area: float = 0.0
    # Geometric reachability volume approximation
    geometric_reachability_volume: float = 0.0
    # DeepMill Diagnostic Data (Non-Contract) - NO INTERPRETATION
    deepmill_diagnostics: Any = None
    # BrepMFR Diagnostic Data (DEPRECATED) - Use graph_feature_diagnostics
    brep_mfr_diagnostics: Any = None
    # Graph-Native Feature Diagnostics (Non-Authoritative)
    graph_feature_diagnostics: Any = None
    # Canonical B-Rep Graph for this setup's reachable faces
    brep_graph: Optional[Any] = None  # Optional[BRepGraph] - avoid circular import

    def summary(self) -> str:
        # Explicit naming in output
        base_str = (f"Setup Direction: ({self.orientation_vector[0]:.2f}, {self.orientation_vector[1]:.2f}, {self.orientation_vector[2]:.2f}) | "
                f"Tool: {self.tool_diameter}mm | "
                f"Reachable Faces (Count): {len(self.reachable_face_ids)} | "
                f"Reachable Surface Area (Geo): {self.geometric_reachable_surface_area:.2f} mm^2 | "
                f"Reachability Volume (Geo): {self.geometric_reachability_volume:.2f} mm^3 | "
                f"Aligned Area: {self.geometric_aligned_surface_area:.2f} mm^2 | "
                f"Reachability Volume (Geo): {self.geometric_reachability_volume:.2f} mm^3 | "
                f"Aligned Area: {self.geometric_aligned_surface_area:.2f} mm^2 | "
                f"Perp Area: {self.geometric_perpendicular_surface_area:.2f} mm^2")
        
        # DeepMill Diagnostics Block
        if self.deepmill_diagnostics is not None:
            diag_str = "Captured (Unknown Format)"
            try:
                # Introspection for Tensor/Array-like objects without hard dependency
                d = self.deepmill_diagnostics
                type_name = type(d).__name__
                
                if 'Tensor' in type_name or 'ndarray' in type_name:
                    shape = getattr(d, 'shape', 'Unknown')
                    # Attempt basic stats if available
                    stats = []
                    if hasattr(d, 'min'): stats.append(f"Min:{d.min():.2f}")
                    if hasattr(d, 'max'): stats.append(f"Max:{d.max():.2f}")
                    if hasattr(d, 'mean'): stats.append(f"Mean:{d.mean():.2f}")
                    
                    diag_str = f"[{type_name}] Shape:{shape} Stats:({' '.join(stats)})"
                else:
                    diag_str = f"Captured ({type_name})"
            except Exception:
                diag_str = "Captured (Introspection Failed)"
            
            # return f"{base_str} | DeepMill Diagnostics: {diag_str}"  <-- REMOVED
            base_str = f"{base_str} | DeepMill Diagnostics: {diag_str}"
        else:
             base_str = f"{base_str} | DeepMill Diagnostics: None"

        # BrepMFR Diagnostics Block
        if self.brep_mfr_diagnostics is not None:
            # Summary: Count of recognized features
            try:
                # self.brep_mfr_diagnostics is Dict[int, Dict]
                count = len(self.brep_mfr_diagnostics)
                base_str = f"{base_str} | BrepMFR Diagnostics: {count} Faces Classified"
            except:
                base_str = f"{base_str} | BrepMFR Diagnostics: Error Reading"
        else:
            base_str = f"{base_str} | BrepMFR Diagnostics: None"
            
        return base_str

class SetupPartitioner:
    """
    Logic to partition geometry into Setups based on accessibility.
    """
    @staticmethod
    def creates_setup_from_accessibility(
        geo_context: Any,
        axis: Tuple[float, float, float],
        tool_diameter: float, # Moved before defaults
        accessibility_ratio: float = 0.0, # From Accessibility Engine
        deepmill_diagnostics: Any = None, # Diagnostic metadata
        brep_mfr_diagnostics: Any = None # BrepMFR metadata
    ) -> SetupContext:
        """
        Creates a SetupContext for the given axis.
        Rule: A face is considered accessible if its normal at the center
        has a positive dot product with the approach direction (facing the tool).
        """
        setup = SetupContext(
            orientation_vector=axis,
            tool_diameter=tool_diameter,
            deepmill_diagnostics=deepmill_diagnostics,
            brep_mfr_diagnostics=brep_mfr_diagnostics
        )
        
        # Compute Geometric Reachability Volume
        # (Total Volume * Accessibility Ratio)
        if hasattr(geo_context, 'total_volume'):
             setup.geometric_reachability_volume = geo_context.total_volume * accessibility_ratio
        
        # Iterate through STEP faces in GeometryContext
        # Note: This relies on geo_context attributes established in analysis_engine/main.py
        if hasattr(geo_context, 'step_face_ids'):
            for face_id, face in geo_context.step_face_ids.items():
                try:
                    # Get normal at center of UV parameter space
                    # This is an approximation for complex surfaces but deterministic
                    center_uv = face.Center()
                    normal_vec = face.normalAt(center_uv).toTuple()
                    
                    # Dot product
                    dot = (normal_vec[0] * axis[0] + 
                           normal_vec[1] * axis[1] + 
                           normal_vec[2] * axis[2])
                    
                    # Tolerance for "facing the tool". 
                    # > 0 means generally facing.
                    if dot > 0.0:
                        setup.reachable_face_ids.append(face_id)
                        
                        # Compute Area
                        gprops = GProp_GProps()
                        BRepGProp.SurfaceProperties_s(face.wrapped, gprops)
                        area = gprops.Mass()
                        setup.geometric_reachable_surface_area += area
                        
                        # Orientation Classification (Geometric Fact)
                        # Threshold 0.707 (approx 45 degrees)
                        if dot >= 0.707:
                            setup.geometric_aligned_surface_area += area
                        else:
                            setup.geometric_perpendicular_surface_area += area
                        
                except:
                    # Robustness for faces where normal/area calc might fail
                    pass
                    
        return setup
