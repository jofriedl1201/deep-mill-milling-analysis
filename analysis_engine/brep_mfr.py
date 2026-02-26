"""
Graph-Native Feature Recognition Module

This module provides diagnostic feature recognition directly on BRepGraph structures.
It operates purely on the canonical graph representation, not raw OCC geometry.

> [!IMPORTANT]
> This is a DIAGNOSTIC module only. It provides structural observations, not
> authoritative machining feature classifications. All outputs are labeled as
> "diagnostic" and must not be used for automated decision-making.

## BrepMFR Status
The original BrepMFR model (https://github.com/zhangshuming0668/BrepMFR) is a RESEARCH
REFERENCE only. It is NOT currently executable due to:
- Missing preprocessing code in public repository
- No pretrained checkpoints available
- DGL version incompatibility with current environment

This module implements BrepMFR-inspired encodings but uses rule-based heuristics
instead of neural network inference.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureDiagnostic:
    """
    Diagnostic observation about a face's potential manufacturing feature.
    
    NOT an authoritative classification - diagnostic only.
    """
    face_id: int
    surface_type: str
    diagnostic_label: str
    confidence: str  # "high", "medium", "low"
    reasoning: str
    adjacent_count: int
    convex_edge_count: int
    concave_edge_count: int


class GraphNativeFeatureRecognizer:
    """
    Diagnostic feature recognition operating directly on BRepGraph.
    
    This replaces the placeholder BrepMFRInference class. It uses rule-based
    heuristics derived from the graph topology to provide structural observations.
    
    All outputs are explicitly labeled as DIAGNOSTIC and non-authoritative.
    """
    
    SURFACE_TYPE_NAMES = {
        0: "Plane",
        1: "Cylinder", 
        2: "Cone",
        3: "Sphere",
        4: "Torus",
        5: "Bezier",
        6: "BSpline",
        7: "Other"
    }
    
    def __init__(self):
        """Initialize the graph-native feature recognizer."""
        pass
    
    def analyze(self, graph: Any) -> Dict[int, FeatureDiagnostic]:
        """
        Analyzes a BRepGraph and produces diagnostic feature observations.
        
        Args:
            graph: BRepGraph instance (imported type avoided for flexibility)
            
        Returns:
            Dict mapping face_id -> FeatureDiagnostic
        """
        results = {}
        
        # Pre-compute edge convexity maps
        convex_faces = {}  # face_id -> count of convex edges
        concave_faces = {}  # face_id -> count of concave edges
        
        for edge_data in graph.edges.values():
            f1, f2 = edge_data.get("face_ids", (-1, -1))
            convexity = edge_data.get("convexity", 2)
            
            if convexity == 1:  # Convex
                convex_faces[f1] = convex_faces.get(f1, 0) + 1
                convex_faces[f2] = convex_faces.get(f2, 0) + 1
            elif convexity == 0:  # Concave
                concave_faces[f1] = concave_faces.get(f1, 0) + 1
                concave_faces[f2] = concave_faces.get(f2, 0) + 1
        
        for face_id, face_data in graph.faces.items():
            surface_type = face_data.get("surface_type", 7)
            surface_name = self.SURFACE_TYPE_NAMES.get(surface_type, "Unknown")
            adjacent_count = face_data.get("adjacency_count", 0)
            area = face_data.get("area", 0.0)
            
            convex_count = convex_faces.get(face_id, 0)
            concave_count = concave_faces.get(face_id, 0)
            
            # Apply rule-based heuristics
            diagnostic_label, confidence, reasoning = self._classify_face(
                surface_type, adjacent_count, convex_count, concave_count, area
            )
            
            results[face_id] = FeatureDiagnostic(
                face_id=face_id,
                surface_type=surface_name,
                diagnostic_label=diagnostic_label,
                confidence=confidence,
                reasoning=reasoning,
                adjacent_count=adjacent_count,
                convex_edge_count=convex_count,
                concave_edge_count=concave_count
            )
        
        return results
    
    def _classify_face(
        self, 
        surface_type: int, 
        adjacent_count: int,
        convex_count: int,
        concave_count: int,
        area: float
    ) -> tuple:
        """
        Rule-based classification heuristics.
        
        Returns: (diagnostic_label, confidence, reasoning)
        """
        # Planar face rules
        if surface_type == 0:  # Plane
            if concave_count >= 4 and adjacent_count >= 4:
                return ("Potential Pocket Floor", "medium", 
                        f"Planar with {concave_count} concave edges suggests pocket floor")
            elif convex_count >= 3 and adjacent_count <= 4:
                return ("Potential Step Surface", "low",
                        f"Planar with {convex_count} convex edges suggests step")
            else:
                return ("Stock Face / Datum", "low",
                        "Planar face, likely stock or datum surface")
        
        # Cylindrical face rules
        if surface_type == 1:  # Cylinder
            if adjacent_count == 2:
                return ("Potential Hole Wall", "high",
                        "Cylindrical with 2 adjacencies suggests through hole")
            elif adjacent_count == 1:
                return ("Potential Blind Hole Wall", "medium",
                        "Cylindrical with 1 adjacency suggests blind hole")
            elif concave_count >= 2:
                return ("Potential Fillet/Round", "medium",
                        f"Cylindrical with {concave_count} concave edges suggests fillet")
            else:
                return ("Cylindrical Feature", "low",
                        "Cylindrical surface, classification unclear")
        
        # Cone rules
        if surface_type == 2:  # Cone
            return ("Potential Chamfer/Countersink", "medium",
                    "Conical surface often indicates chamfer or countersink")
        
        # Sphere rules
        if surface_type == 3:  # Sphere
            return ("Potential Ball End Feature", "low",
                    "Spherical surface, possibly ball end mill trace")
        
        # Torus rules
        if surface_type == 4:  # Torus
            return ("Potential Fillet/Blend", "medium",
                    "Toroidal surface typically indicates fillet or blend")
        
        # Complex surfaces
        if surface_type in (5, 6):  # Bezier, BSpline
            return ("Complex Sculptured Surface", "low",
                    "Freeform surface requires 5-axis consideration")
        
        return ("Unclassified", "low", "Surface type not recognized")
    
    def summary(self, diagnostics: Dict[int, FeatureDiagnostic]) -> str:
        """
        Generate a human-readable summary of diagnostics.
        """
        label_counts = {}
        for diag in diagnostics.values():
            label = diag.diagnostic_label
            label_counts[label] = label_counts.get(label, 0) + 1
        
        lines = ["=== Graph-Native Feature Diagnostics (NON-AUTHORITATIVE) ==="]
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {label}: {count} faces")
        lines.append(f"Total: {len(diagnostics)} faces analyzed")
        
        return "\n".join(lines)


# Legacy alias for backwards compatibility
# DEPRECATED: Use GraphNativeFeatureRecognizer instead
class BrepMFRInference:
    """
    DEPRECATED: This class is a compatibility shim.
    
    BrepMFR model inference is NOT available. Use GraphNativeFeatureRecognizer
    for graph-based diagnostic feature analysis.
    """
    
    def __init__(self):
        self._recognizer = GraphNativeFeatureRecognizer()
        self.is_loaded = True
        print("[WARNING] BrepMFRInference is DEPRECATED. Using GraphNativeFeatureRecognizer.")
    
    def predict_features(self, geometry_context: Any, face_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Legacy interface - wraps GraphNativeFeatureRecognizer.
        
        Returns mock-style output for backwards compatibility.
        """
        # Cannot run without graph - return placeholder
        results = {}
        for fid in face_ids:
            results[fid] = {
                "class": "Unknown (Graph Required)",
                "confidence": 0.0,
                "raw_embedding": [0.0] * 5,
                "_deprecated_warning": "Use GraphNativeFeatureRecognizer with BRepGraph"
            }
        return results
