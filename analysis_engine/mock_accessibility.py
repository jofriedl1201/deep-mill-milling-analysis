from typing import Tuple, Any
from analysis_engine.accessibility_contract import AccessibilityQueryEngine, AccessibilityResult, AccessibilityLabel

class MockAccessibilityEngine(AccessibilityQueryEngine):
    """
    Deterministic Mock implementation to verify the pipeline contract.
    Rule: 
    - +Z axis is 100% Accessible.
    - -Z axis is 0% Accessible (100% Occluded).
    - All other axes are 50% Accessible, 50% Occluded.
    - Inaccessible volume is always 0 for this mock.
    """
    def predict(self, 
                geometry_context: Any, 
                tool_axis: Tuple[float, float, float], 
                tool_diameter: float) -> AccessibilityResult:
        
        result = AccessibilityResult(axis_direction=tool_axis)
        result.total_volume = 1000.0
        
        # Simple Deterministic Rule based on Z component
        z_component = tool_axis[2]
        
        if z_component > 0.99: # +Z
            acc_ratio = 1.0
        elif z_component < -0.99: # -Z
            acc_ratio = 0.0
        else:
            acc_ratio = 0.5
            
        result.accessible_volume = result.total_volume * acc_ratio
        result.occluded_volume = result.total_volume * (1.0 - acc_ratio)
        result.inaccessible_volume = 0.0 # Simplify for mock
        
        return result
