"""
B-Rep Graph Representation Module

This module provides the canonical geometric abstraction for machining feature reasoning.
All downstream analysis should consume this typed container, not raw Open Cascade geometry.

See GRAPH_SCHEMA.md for full specification.
"""

import sys
import os
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add BREP Graph Generator to path
_BREP_GEN_PATH = os.path.join(os.path.dirname(__file__), "..", "BREP Graph Generator")
if os.path.exists(_BREP_GEN_PATH):
    sys.path.insert(0, _BREP_GEN_PATH)


@dataclass
class BRepGraph:
    """
    Canonical B-Rep graph representation for machining feature reasoning.
    
    This is a pure-data container with no ML framework dependencies.
    All tensors are numpy arrays.
    
    Attributes:
        faces: Dict mapping face_id -> face feature dict
        edges: Dict mapping edge_id -> edge feature dict
        spatial_pos: (N, N) shortest path distances
        edge_path: (N, N, 16) edge indices along paths
        d2_distance: (N, N, 64) D2 shape distribution histograms
        angle_distance: (N, N, 64) A3 angular histograms
        source_file: Original STEP file path (for provenance)
        graph_hash: Determinism verification hash
    """
    faces: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    edges: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    spatial_pos: np.ndarray = field(default_factory=lambda: np.array([]))
    edge_path: np.ndarray = field(default_factory=lambda: np.array([]))
    d2_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    angle_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    source_file: str = ""
    graph_hash: str = ""
    
    @property
    def num_faces(self) -> int:
        return len(self.faces)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)
    
    def get_face_ids(self) -> List[int]:
        """Returns sorted list of face IDs."""
        return sorted(self.faces.keys())
    
    def get_adjacent_faces(self, face_id: int) -> List[int]:
        """Returns list of faces adjacent to the given face."""
        if face_id not in self.faces:
            return []
        return self.faces[face_id].get("adjacent_faces", [])
    
    def get_faces_by_type(self, surface_type: int) -> List[int]:
        """Returns face IDs matching the given surface type."""
        return [fid for fid, fdata in self.faces.items() 
                if fdata.get("surface_type") == surface_type]
    
    def get_convex_edges(self) -> List[int]:
        """Returns edge IDs that are convex (convexity=1)."""
        return [eid for eid, edata in self.edges.items() 
                if edata.get("convexity") == 1]
    
    def get_concave_edges(self) -> List[int]:
        """Returns edge IDs that are concave (convexity=0)."""
        return [eid for eid, edata in self.edges.items() 
                if edata.get("convexity") == 0]
    
    def summary(self) -> str:
        """Returns a human-readable summary of the graph."""
        type_counts = {}
        for fdata in self.faces.values():
            st = fdata.get("surface_type", -1)
            type_counts[st] = type_counts.get(st, 0) + 1
        
        type_names = {0: "Plane", 1: "Cylinder", 2: "Cone", 3: "Sphere", 
                      4: "Torus", 5: "Bezier", 6: "BSpline", 7: "Other"}
        
        type_str = ", ".join([f"{type_names.get(k, 'Unknown')}:{v}" 
                              for k, v in sorted(type_counts.items())])
        
        return (f"BRepGraph: {self.num_faces} faces, {self.num_edges} edges | "
                f"Types: [{type_str}] | Hash: {self.graph_hash[:8] if self.graph_hash else 'N/A'}")


class BRepGraphGenerator:
    """
    Generates BRepGraph from STEP files using the fast-path analyzer.
    
    This is the single entry point for B-Rep graph creation.
    It wraps the low-level brep_step1.analyze_step_faces_fast function.
    """
    
    def __init__(self, use_fast_path: bool = True, time_budget_sec: float = 5.0):
        """
        Args:
            use_fast_path: If True, use bounded-time fast path. If False, use training-grade.
            time_budget_sec: Time budget for fast path.
        """
        self.use_fast_path = use_fast_path
        self.time_budget_sec = time_budget_sec
        self._analyzer = None
    
    def _ensure_analyzer(self):
        """Lazy import of analyzer to avoid import errors if not available."""
        if self._analyzer is None:
            try:
                from brep_step1 import analyze_step_faces_fast, analyze_step_faces
                self._analyzer = {
                    "fast": analyze_step_faces_fast,
                    "full": analyze_step_faces
                }
            except ImportError as e:
                raise ImportError(
                    f"Cannot import brep_step1 module. "
                    f"Ensure 'BREP Graph Generator' folder is in path. Error: {e}"
                )
    
    def generate(self, step_path: str) -> BRepGraph:
        """
        Generates a BRepGraph from a STEP file.
        
        Args:
            step_path: Path to STEP file.
            
        Returns:
            BRepGraph containing all face, edge, and global features.
        """
        self._ensure_analyzer()
        
        # Select analyzer
        if self.use_fast_path:
            raw_data = self._analyzer["fast"](step_path, time_budget_sec=self.time_budget_sec)
        else:
            raw_data = self._analyzer["full"](step_path)
        
        # Compute hash for determinism verification
        hash_input = (
            str(len(raw_data["faces"])) + 
            str(raw_data["spatial_pos"].tobytes()[:100])
        )
        graph_hash = hashlib.md5(hash_input.encode()).hexdigest()
        
        # Construct BRepGraph
        graph = BRepGraph(
            faces=raw_data["faces"],
            edges=raw_data["edges"],
            spatial_pos=raw_data["spatial_pos"],
            edge_path=raw_data["edge_path"],
            d2_distance=raw_data["d2_distance"],
            angle_distance=raw_data["angle_distance"],
            source_file=step_path,
            graph_hash=graph_hash
        )
        
        return graph
    
    def validate_determinism(self, step_path: str, runs: int = 3) -> Tuple[bool, str]:
        """
        Validates that graph generation is deterministic.
        
        Args:
            step_path: Path to STEP file.
            runs: Number of runs to compare.
            
        Returns:
            (is_deterministic, message)
        """
        hashes = []
        for i in range(runs):
            graph = self.generate(step_path)
            hashes.append(graph.graph_hash)
        
        is_deterministic = len(set(hashes)) == 1
        if is_deterministic:
            return True, f"Deterministic: {runs} runs produced hash {hashes[0][:8]}"
        else:
            return False, f"Non-deterministic: hashes differ {hashes}"


def extract_subgraph(graph: BRepGraph, face_ids: List[int]) -> BRepGraph:
    """
    Extracts a subgraph containing only the specified faces.
    
    Useful for per-setup analysis where only reachable faces are relevant.
    
    Args:
        graph: Source BRepGraph
        face_ids: List of face IDs to include
        
    Returns:
        New BRepGraph with only the specified faces and their connecting edges.
    """
    face_id_set = set(face_ids)
    
    # Filter faces
    new_faces = {fid: fdata for fid, fdata in graph.faces.items() if fid in face_id_set}
    
    # Filter edges (both endpoints must be in subset)
    new_edges = {}
    for eid, edata in graph.edges.items():
        f1, f2 = edata.get("face_ids", (-1, -1))
        if f1 in face_id_set and f2 in face_id_set:
            new_edges[eid] = edata
    
    # Create index mapping for subgraph matrices
    sorted_ids = sorted(face_ids)
    id_map = {old_id: new_idx for new_idx, old_id in enumerate(sorted_ids)}
    n = len(sorted_ids)
    
    # Extract submatrices
    new_spatial_pos = np.full((n, n), 256, dtype=np.int32)
    new_d2_distance = np.zeros((n, n, 64), dtype=np.int32)
    new_angle_distance = np.zeros((n, n, 64), dtype=np.int32)
    new_edge_path = np.full((n, n, 16), -1, dtype=np.int32)
    
    for old_i, new_i in id_map.items():
        for old_j, new_j in id_map.items():
            if old_i < graph.spatial_pos.shape[0] and old_j < graph.spatial_pos.shape[1]:
                new_spatial_pos[new_i, new_j] = graph.spatial_pos[old_i, old_j]
                new_d2_distance[new_i, new_j] = graph.d2_distance[old_i, old_j]
                new_angle_distance[new_i, new_j] = graph.angle_distance[old_i, old_j]
                new_edge_path[new_i, new_j] = graph.edge_path[old_i, old_j]
    
    return BRepGraph(
        faces=new_faces,
        edges=new_edges,
        spatial_pos=new_spatial_pos,
        edge_path=new_edge_path,
        d2_distance=new_d2_distance,
        angle_distance=new_angle_distance,
        source_file=graph.source_file,
        graph_hash=f"subgraph_{len(face_ids)}_of_{graph.graph_hash[:8]}"
    )
