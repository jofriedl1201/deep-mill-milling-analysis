from enum import Enum, auto
from typing import Optional, Tuple, Any

class AccessibilityLabel(Enum):
    """
    Formal classification for region accessibility.
    """
    ACCESSIBLE = auto()  # Reachable and machinable
    INACCESSIBLE = auto() # Physically impossible for tool size (radius violation)
    OCCLUDED = auto()     # Blocked from specific direction (reach/shadowing violation)
    UNKNOWN = auto()      # Not evaluated

class AccessibilityResult:
    """
    Standardized output of an Accessibility Query.
    Currently assumes a volumetric or aggregate representation.
    """
    def __init__(self, axis_direction: Tuple[float, float, float]):
        self.axis_direction = axis_direction
        self.total_volume = 0.0          # Conceptual Total Volume
        self.accessible_volume = 0.0     # Volume marked ACCESSIBLE
        self.inaccessible_volume = 0.0   # Volume marked INACCESSIBLE
        self.occluded_volume = 0.0       # Volume marked OCCLUDED
    
    @property
    def accessibility_ratio(self) -> float:
        """Percentage of volume that is accessible from this direction."""
        if self.total_volume <= 0:
            return 0.0
        return self.accessible_volume / self.total_volume

    def summary(self) -> str:
        return (f"Axis({self.axis_direction[0]:.2f}, {self.axis_direction[1]:.2f}, {self.axis_direction[2]:.2f}) "
                f"-> Acc: {self.accessibility_ratio:.1%} "
                f"(Vol: {self.accessible_volume:.2f}/{self.total_volume:.2f})")
