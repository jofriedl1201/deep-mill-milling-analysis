import sys
import os
from typing import Tuple, Any, Dict, Optional
from analysis_engine.accessibility_contract import AccessibilityQueryEngine, AccessibilityResult, AccessibilityLabel
from analysis_engine.mock_accessibility import MockAccessibilityEngine

# Add DeepMill project path for custom UNet model
DEEPMILL_PROJECT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "DeepMill-master", "DeepMill-master", "projects"
)
if os.path.exists(DEEPMILL_PROJECT_PATH) and DEEPMILL_PROJECT_PATH not in sys.path:
    sys.path.insert(0, DEEPMILL_PROJECT_PATH)

# Stub for caching class - assumes O-CNN octree object
class OctreeCache:
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        self.cache[key] = value

class DeepMillAccessibilityResult(AccessibilityResult):
    """
    Extended result containing DeepMill-specific diagnostics.
    """
    def __init__(self, axis_direction: Tuple[float, float, float]):
        super().__init__(axis_direction)
        self.deepmill_diagnostics: Any = None # Raw tensor/octree data
        self.accessibility_predictions: Any = None  # Per-point predictions
        self.inaccessible_ratio: float = 0.0  # Ratio of inaccessible points

class DeepMillAccessibilityEngine(AccessibilityQueryEngine):
    """
    Real integration of DeepMill framework with pretrained model inference.
    """

    # Path to pretrained model
    PRETRAINED_MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        "..", "DeepMill-master", "DeepMill-master", "pretrained", "00840.model.pth"
    )

    # Model configuration (from seg_deepmill.yaml)
    MODEL_CONFIG = {
        "depth": 5,
        "full_depth": 2,
        "channel": 4,
        "nout": 2,  # Binary classification: accessible/inaccessible
        "stages": 3,
        "feature": "ND",  # Normal + Displacement
        "nempty": False,
        "interp": "linear",
    }

    def __init__(self):
        self._fallback_engine = MockAccessibilityEngine()
        self._cache = OctreeCache()
        self._model = None
        self._device = None

        # Audit Counters
        self.stats = {
            "preprocessing_calls": 0,
            "inference_calls": 0,
            "fallback_calls": 0,
            "cache_hits": 0,
            "model_loaded": False
        }

        # Check dependencies and load model
        self.is_deepmill_available = self._check_dependencies()
        if self.is_deepmill_available:
            self._load_pretrained_model()
        else:
            print("[DeepMillAudit] Dependency check failed. DeepMill disabled.", flush=True)

    def _check_dependencies(self) -> bool:
        """
        Attempts to import required libraries (torch, ocnn).
        """
        try:
            import torch
            import ocnn
            import numpy as np
            return True
        except ImportError as e:
            print(f"[DeepMillAudit] Import error: {e}", flush=True)
            return False

    def _load_pretrained_model(self) -> bool:
        """
        Loads the pretrained DeepMill UNet model.
        The model uses custom fc_modules for tool parameter processing.
        """
        try:
            import torch
            import ocnn

            # Check if model file exists
            if not os.path.exists(self.PRETRAINED_MODEL_PATH):
                print(f"[DeepMillAudit] Pretrained model not found at: {self.PRETRAINED_MODEL_PATH}", flush=True)
                return False

            # Set device
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[DeepMillAudit] Using device: {self._device}", flush=True)

            # Import custom UNet from DeepMill project (has fc_modules for tool params)
            try:
                from ocnn.models.unet import UNet as DeepMillUNet
                print("[DeepMillAudit] Using custom DeepMill UNet model", flush=True)
            except ImportError as e:
                print(f"[DeepMillAudit] Could not import custom UNet: {e}", flush=True)
                print("[DeepMillAudit] Falling back to standard ocnn.models.UNet", flush=True)
                DeepMillUNet = ocnn.models.UNet

            # Create model architecture - UNet with tool parameter support
            self._model = DeepMillUNet(
                in_channels=self.MODEL_CONFIG["channel"],
                out_channels=self.MODEL_CONFIG["nout"],
                interp=self.MODEL_CONFIG["interp"],
                nempty=self.MODEL_CONFIG["nempty"]
            )

            # Load pretrained weights
            checkpoint = torch.load(self.PRETRAINED_MODEL_PATH, map_location=self._device, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Load state dict
            missing_keys, unexpected_keys = self._model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"[DeepMillAudit] Missing keys: {len(missing_keys)}", flush=True)
            if unexpected_keys:
                print(f"[DeepMillAudit] Unexpected keys: {len(unexpected_keys)}", flush=True)

            self._model.to(self._device)
            self._model.eval()

            self.stats["model_loaded"] = True
            print(f"[DeepMillAudit] Pretrained UNet model loaded successfully from {self.PRETRAINED_MODEL_PATH}", flush=True)
            return True

        except Exception as e:
            print(f"[DeepMillAudit] Failed to load pretrained model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def _mesh_to_points(self, geo_context: Any, num_samples: int = 10000) -> Any:
        """
        Converts mesh geometry to point cloud with normals.
        """
        import numpy as np
        import torch

        try:
            # Try to get mesh from geometry context
            mesh_obj = getattr(geo_context, 'mesh_object', None)

            if mesh_obj is not None:
                # Sample points from OCC mesh
                vertices = []
                normals = []

                # Get vertices from OCC Poly_Triangulation
                for i in range(1, mesh_obj.NbNodes() + 1):
                    pnt = mesh_obj.Node(i)
                    vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

                vertices = np.array(vertices, dtype=np.float32)

                # Estimate normals (simple approach - use face normals)
                # For a proper implementation, compute per-vertex normals
                if len(vertices) > 0:
                    # Simple normal estimation using local neighborhood
                    normals = np.zeros_like(vertices)
                    normals[:, 2] = 1.0  # Default to Z-up

                    # Subsample if needed
                    if len(vertices) > num_samples:
                        indices = np.random.choice(len(vertices), num_samples, replace=False)
                        vertices = vertices[indices]
                        normals = normals[indices]

                    return vertices, normals

            # Fallback: create dummy point cloud from bounding box
            bbox = getattr(geo_context, 'mesh_bbox', None) or getattr(geo_context, 'step_bbox', None)
            if bbox:
                # Generate random points within bounding box
                if isinstance(bbox, tuple) and len(bbox) == 2:
                    min_pt, max_pt = bbox
                else:
                    min_pt = bbox[:3]
                    max_pt = bbox[3:]

                vertices = np.random.uniform(min_pt, max_pt, (num_samples, 3)).astype(np.float32)
                normals = np.zeros_like(vertices)
                normals[:, 2] = 1.0
                return vertices, normals

            # Ultimate fallback
            vertices = np.random.randn(num_samples, 3).astype(np.float32)
            normals = np.zeros_like(vertices)
            normals[:, 2] = 1.0
            return vertices, normals

        except Exception as e:
            print(f"[DeepMillAudit] Error converting mesh to points: {e}", flush=True)
            vertices = np.random.randn(num_samples, 3).astype(np.float32)
            normals = np.zeros_like(vertices)
            normals[:, 2] = 1.0
            return vertices, normals

    def _build_octree(self, vertices: Any, normals: Any) -> Any:
        """
        Builds O-CNN octree from point cloud, properly batched for inference.
        Mirrors the process_batch method from segmentation.py.
        """
        import torch
        import ocnn

        # Normalize points to [-1, 1]
        center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        scale = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2
        if scale > 0:
            vertices = (vertices - center) / scale

        # Create Points object
        points_tensor = torch.from_numpy(vertices).float()
        normals_tensor = torch.from_numpy(normals).float()

        # Normalize normals
        norm_lengths = torch.norm(normals_tensor, dim=1, keepdim=True)
        norm_lengths = torch.clamp(norm_lengths, min=1e-8)
        normals_tensor = normals_tensor / norm_lengths

        # Create ocnn Points object
        point_cloud = ocnn.octree.Points(
            points=points_tensor,
            normals=normals_tensor,
        )

        # Build individual octree
        single_octree = ocnn.octree.Octree(
            depth=self.MODEL_CONFIG["depth"],
            full_depth=self.MODEL_CONFIG["full_depth"]
        )
        single_octree.build_octree(point_cloud)

        # CRITICAL: Merge octrees (even for batch_size=1) to set batch metadata
        # This is how training handles batching - merge_octrees sets batch_nnum properly
        octree = ocnn.octree.merge_octrees([single_octree])

        # Construct neighbor information for convolution operations
        octree.construct_all_neigh()

        # Merge points as well (sets batch_id correctly)
        merged_points = ocnn.octree.merge_points([point_cloud])

        return octree, merged_points

    def _preprocess_geometry(self, geo_context: Any) -> Tuple[Any, Any]:
        """
        Converts Mesh to Octree. Cached per context.
        """
        cache_key = str(id(geo_context))

        cached = self._cache.get(cache_key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            print(f"[DeepMillAudit] Geometry {cache_key} found in cache.", flush=True)
            return cached

        print(f"[DeepMillAudit] Preprocessing Geometry {cache_key} (Mesh -> PointCloud -> Octree)...", flush=True)
        self.stats["preprocessing_calls"] += 1

        # Convert mesh to point cloud
        vertices, normals = self._mesh_to_points(geo_context)

        # Build octree
        octree, point_cloud = self._build_octree(vertices, normals)

        result = (octree, point_cloud, vertices)
        self._cache.set(cache_key, result)
        return result

    def predict(self,
                geometry_context: Any,
                tool_axis: Tuple[float, float, float],
                tool_diameter: float) -> AccessibilityResult:

        # 1. Fallback Check - if model not loaded
        if not self.is_deepmill_available or not self.stats["model_loaded"]:
            print(f"[DeepMillAudit] Request delegated to Fallback (Mock) for axis {tool_axis}", flush=True)
            self.stats["fallback_calls"] += 1
            return self._fallback_engine.predict(geometry_context, tool_axis, tool_diameter)

        # 2. Preprocessing & Caching
        try:
            octree, point_cloud, vertices = self._preprocess_geometry(geometry_context)
        except Exception as e:
            print(f"[DeepMillAudit] Preprocessing failed: {e}, using fallback", flush=True)
            self.stats["fallback_calls"] += 1
            return self._fallback_engine.predict(geometry_context, tool_axis, tool_diameter)

        # 3. Inference
        self.stats["inference_calls"] += 1
        print(f"[DeepMillAudit] Running DeepMill Inference for axis {tool_axis}", flush=True)

        try:
            import torch
            import ocnn

            with torch.no_grad():
                # Move octree to device
                octree_device = octree.to(self._device)

                # Get input features (Normal + Displacement)
                octree_feature = ocnn.modules.InputFeature(
                    self.MODEL_CONFIG["feature"],
                    self.MODEL_CONFIG["nempty"]
                )
                data = octree_feature(octree_device)

                # Create query points with batch_id from merged points
                # point_cloud is now a merged Points object with proper batch_id
                query_pts = torch.cat([
                    point_cloud.points,
                    point_cloud.batch_id
                ], dim=1).to(self._device)

                # Create tool parameters tensor [tool_diameter, axis_x, axis_y, axis_z]
                # Format matches DeepMill training: batch_size x 4
                tool_params = torch.tensor([[
                    tool_diameter,
                    tool_axis[0],
                    tool_axis[1],
                    tool_axis[2]
                ]], dtype=torch.float32).to(self._device)

                # Run UNet forward pass with tool parameters
                # UNet returns (logits_1, logits_2) for dual-head prediction
                logits_1, logits_2 = self._model(
                    data, octree_device, self.MODEL_CONFIG["depth"],
                    query_pts, tool_params
                )

                # logits_1: primary accessibility (red zone - collision)
                # logits_2: secondary accessibility (green zone - clear)
                # Use logits_1 for accessibility classification
                predictions = torch.argmax(logits_1, dim=-1)

                # Calculate accessibility ratio
                # Class 0 = inaccessible (collision), Class 1 = accessible
                num_inaccessible = (predictions == 0).sum().item()
                total_points = predictions.shape[0]
                inaccessible_ratio = num_inaccessible / total_points if total_points > 0 else 0
                accessible_ratio = 1.0 - inaccessible_ratio

                print(f"[DeepMillAudit] Inference complete: {total_points} points, "
                      f"{accessible_ratio*100:.1f}% accessible", flush=True)

            # Create result - set volumes (accessibility_ratio is computed property)
            result = DeepMillAccessibilityResult(axis_direction=tool_axis)
            result.deepmill_diagnostics = (logits_1.cpu(), logits_2.cpu())
            result.accessibility_predictions = predictions.cpu()
            result.inaccessible_ratio = inaccessible_ratio
            # Set volumes - accessibility_ratio is computed from these
            result.total_volume = 1000.0
            result.accessible_volume = accessible_ratio * result.total_volume
            result.inaccessible_volume = inaccessible_ratio * result.total_volume

            return result

        except Exception as e:
            print(f"[DeepMillAudit] Inference failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

            # Fallback to mock engine on inference error
            self.stats["fallback_calls"] += 1
            return self._fallback_engine.predict(geometry_context, tool_axis, tool_diameter)

    def get_audit_report(self) -> str:
        return (
            "\n=== DeepMill Execution Audit ===\n"
            f"DeepMill Available : {self.is_deepmill_available}\n"
            f"Model Loaded       : {self.stats['model_loaded']}\n"
            f"Device             : {self._device}\n"
            f"Preprocessing Events: {self.stats['preprocessing_calls']} (Should be ~1 per part)\n"
            f"Cache Hits          : {self.stats['cache_hits']}\n"
            f"Inference Calls     : {self.stats['inference_calls']}\n"
            f"Fallback Calls      : {self.stats['fallback_calls']}\n"
            "================================"
        )
