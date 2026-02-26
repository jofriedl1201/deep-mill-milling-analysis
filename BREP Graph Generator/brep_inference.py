import sys
import os
import torch
import numpy as np
import argparse

# ============================================================================
# DEPENDENCY DETECTION
# ============================================================================

# Check for DGL
try:
    import dgl
    DGL_INSTALLED = True
except ImportError:
    DGL_INSTALLED = False
    dgl = None

# Dynamic import setup for BrepMFR
BREP_MFR_PATH = os.path.join(os.path.dirname(__file__), "..", "BrepMFR")
if os.path.exists(BREP_MFR_PATH):
    sys.path.append(BREP_MFR_PATH)
else:
    BREP_MFR_PATH = os.path.join(os.getcwd(), "BrepMFR")
    if os.path.exists(BREP_MFR_PATH):
        sys.path.append(BREP_MFR_PATH)

try:
    from models.brepseg_model import BrepSeg
    from data.collator import collator
    BREP_MFR_AVAILABLE = True
except ImportError:
    BrepSeg = None
    collator = None
    BREP_MFR_AVAILABLE = False


class PYGGraph:
    """Mock PyG Data object to match dataset.py expectations."""
    def __init__(self):
        self.graph = None
        self.node_data = None
        self.label_feature = None


def check_inference_capability(checkpoint_path=None):
    """
    Determines whether ML inference can run.
    
    Returns:
        tuple: (inference_available: bool, reason: str)
    """
    if not DGL_INSTALLED:
        return False, "missing_dgl"
    
    if not BREP_MFR_AVAILABLE:
        return False, "missing_brepMFR"
    
    if checkpoint_path is None:
        return False, "missing_checkpoint"
    
    if not os.path.exists(checkpoint_path):
        return False, "checkpoint_not_found"
    
    return True, "ready"


def run_inference(dgl_graph, aux_tensors, checkpoint_path, device='cpu'):
    """
    Runs BrepMFR inference with graceful degradation.
    
    If dependencies are missing or checkpoint is unavailable, returns
    a structured skip result instead of crashing.
    
    Returns:
        dict: {
            "status": "inference_completed" | "inference_skipped",
            "reason": str (only if skipped),
            "predictions": dict[int, int] | None
        }
    """
    # Check capability
    can_run, reason = check_inference_capability(checkpoint_path)
    
    if not can_run:
        return {
            "status": "inference_skipped",
            "reason": reason,
            "predictions": None
        }
    
    # Run inference
    try:
        # 1. Adapt to PyGGraph structure
        item = PYGGraph()
        item.graph = dgl_graph
        
        # Node features
        item.node_data = dgl_graph.ndata['x'].type(torch.float32)
        item.face_type = dgl_graph.ndata['z'].type(torch.int)
        item.face_area = dgl_graph.ndata['y'].type(torch.float32)
        item.face_loop = dgl_graph.ndata['l'].type(torch.int)
        item.face_adj = dgl_graph.ndata['a'].type(torch.int)
        item.label_feature = dgl_graph.ndata['f'].type(torch.int)
        
        # Edge features
        item.edge_data = dgl_graph.edata['x'].type(torch.float32)
        item.edge_type = dgl_graph.edata['t'].type(torch.int)
        item.edge_len = dgl_graph.edata['l'].type(torch.float32)
        item.edge_ang = dgl_graph.edata['a'].type(torch.float32)
        item.edge_conv = dgl_graph.edata['c'].type(torch.int)
        
        # Graph Topology
        dense_adj = dgl_graph.adj().to_dense().type(torch.int)
        item.node_degree = dense_adj.long().sum(dim=1).view(-1)
        
        # Aux Global Tensors (Direct Mapping)
        item.spatial_pos = aux_tensors['spatial_pos'].type(torch.long)
        item.edge_path = aux_tensors['edge_path'].type(torch.long)
        item.d2_distance = aux_tensors['d2_distance'].type(torch.float32)
        item.angle_distance = aux_tensors['angle_distance'].type(torch.float32)
        
        # Init empty attn_bias (dataset.py does this: torch.zeros([n+1, n+1]))
        n_nodes = dgl_graph.num_nodes()
        item.attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], dtype=torch.float32)
        
        item.data_id = 0
        
        # 2. Collate (Batch size 1, standard params)
        batch = collator([item], multi_hop_max_dist=16, spatial_pos_max=32)
        
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                
        # 3. Load Model
        model = BrepSeg.load_from_checkpoint(checkpoint_path, strict=False)
        model.to(device)
        model.eval()
        
        # 4. Forward
        with torch.no_grad():
            node_emb, graph_emb = model.brep_encoder(batch, last_state_only=True)
            
            node_emb = node_emb[0].permute(1, 0, 2)
            node_emb = node_emb[:, 1:, :] 
            
            padding_mask = batch["padding_mask"]
            node_pos = torch.where(padding_mask == False)
            node_z = node_emb[node_pos]
            
            padding_mask_ = ~padding_mask
            num_nodes_per_graph = torch.sum(padding_mask_.long(), dim=-1)
            
            graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(device)
            
            z = model.attention([node_z, graph_z])
            logits = model.classifier(z)
            preds = torch.argmax(logits, dim=-1)
            
        res = {}
        preds_np = preds.cpu().numpy()
        for i, pred in enumerate(preds_np):
            res[i] = int(pred)
            
        return {
            "status": "inference_completed",
            "predictions": res
        }
        
    except Exception as e:
        # Inference attempted but failed
        return {
            "status": "inference_skipped",
            "reason": f"inference_error: {str(e)}",
            "predictions": None
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    from brep_step1 import analyze_step_faces
    from brep_dgl import assemble_dgl_graph
    
    # Step 1-3: Always run graph extraction
    print("=" * 60)
    print("STEP 1-3: Graph Extraction")
    print("=" * 60)
    analysis = analyze_step_faces(args.step_file)
    print(f"Extracted {len(analysis['faces'])} faces")
    
    # Step 4: Always run DGL assembly
    print("\n" + "=" * 60)
    print("STEP 4: DGL Assembly")
    print("=" * 60)
    g_dgl, aux = assemble_dgl_graph(analysis)
    print(f"DGL Graph: {g_dgl.num_nodes()} nodes, {g_dgl.num_edges()} edges")
    
    # Step 5: Conditionally run inference
    print("\n" + "=" * 60)
    print("STEP 5: ML Inference")
    print("=" * 60)
    
    result = run_inference(g_dgl, aux, args.checkpoint)
    
    if result["status"] == "inference_skipped":
        print(f"⚠ Inference skipped: {result['reason']}")
        print("Graph extraction completed successfully.")
        print("To enable inference, ensure:")
        print("  - DGL is installed (pip install dgl)")
        print("  - BrepMFR modules are available")
        print("  - A valid checkpoint path is provided via --checkpoint")
    else:
        print("✓ Inference completed successfully")
        print("\nPredictions:")
        for fid, cls in result["predictions"].items():
            print(f"  Face {fid}: Class {cls}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed.")
    print("=" * 60)
