import torch
import dgl
import numpy as np

def assemble_dgl_graph(data: dict):
    """
    Converts BrepMFR analysis dictionary to DGL Graph and Aux Tensors.
    (Refactored for Strict Interface Fidelity)
    
    Args:
        data: Output from analyze_step_faces.
        
    Returns:
        tuple: (dgl.DGLGraph, dict)
            - graph: DGL graph with .ndata and .edata
            - aux: Dictionary of global tensors
    """
    faces = data['faces']
    edges = data['edges']
    
    num_nodes = len(faces)
    
    # --- 1. Construct Graph Structure (Bidirectional) ---
    src_list = []
    dst_list = []
    
    # Store indices to map dgl edges back to geometric edge features
    geo_edge_indices = []
    
    # Enumerate edges in deterministic order (by E_ID)
    sorted_e_ids = sorted(edges.keys())
    for e_id in sorted_e_ids:
        edge_info = edges[e_id]
        u, v = edge_info['face_ids']
        
        # Add u->v
        src_list.append(u)
        dst_list.append(v)
        geo_edge_indices.append(e_id)
        
        # Add v->u
        src_list.append(v)
        dst_list.append(u)
        geo_edge_indices.append(e_id)
        
    src_tensor = torch.tensor(src_list, dtype=torch.long)
    dst_tensor = torch.tensor(dst_list, dtype=torch.long)
    
    g = dgl.graph((src_tensor, dst_tensor), num_nodes=num_nodes)
    
    # --- 2. Node Features ---
    # Sort faces by F_ID (should be 0..N-1)
    sorted_f_ids = sorted(faces.keys())
    
    nf_x = [] # uv_grid
    nf_z = [] # surf_type
    nf_y = [] # area
    nf_l = [] # loop_count
    nf_a = [] # adj_count
    nf_f = [] # label (-1)
    
    for f_id in sorted_f_ids:
        f_data = faces[f_id]
        nf_x.append(f_data['uv_grid'])
        nf_z.append(f_data['surface_type'])
        nf_y.append(f_data['area'])
        nf_l.append(f_data['loop_count'])
        nf_a.append(f_data['adjacency_count'])
        nf_f.append(-1)
        
    g.ndata['x'] = torch.tensor(np.array(nf_x), dtype=torch.float32)
    g.ndata['z'] = torch.tensor(np.array(nf_z), dtype=torch.long)
    g.ndata['y'] = torch.tensor(np.array(nf_y), dtype=torch.float32)
    g.ndata['l'] = torch.tensor(np.array(nf_l), dtype=torch.long)
    g.ndata['a'] = torch.tensor(np.array(nf_a), dtype=torch.long)
    g.ndata['f'] = torch.tensor(np.array(nf_f), dtype=torch.long)
    
    # --- 3. Edge Features ---
    ef_x = [] # curve_grid
    ef_t = [] # type
    ef_l = [] # length
    ef_a = [] # angle
    ef_c = [] # convexity
    
    for e_id in geo_edge_indices:
        e_data = edges[e_id]
        ef_x.append(e_data['curve_grid'])
        ef_t.append(e_data['curve_type'])
        ef_l.append(e_data['length'])
        ef_a.append(e_data['dihedral_angle'])
        ef_c.append(e_data['convexity'])
        
    g.edata['x'] = torch.tensor(np.array(ef_x), dtype=torch.float32)
    g.edata['t'] = torch.tensor(np.array(ef_t), dtype=torch.long)
    g.edata['l'] = torch.tensor(np.array(ef_l), dtype=torch.float32)
    g.edata['a'] = torch.tensor(np.array(ef_a), dtype=torch.float32)
    g.edata['c'] = torch.tensor(np.array(ef_c), dtype=torch.long)
    
    # --- 4. Global Tensors (Pass Through) ---
    # Convert numpy to torch
    aux = {
        "spatial_pos": torch.tensor(data['spatial_pos'], dtype=torch.long),
        "edge_path": torch.tensor(data['edge_path'], dtype=torch.long),
        "d2_distance": torch.tensor(data['d2_distance'], dtype=torch.float32), 
        # Note: dataset.py loads d2_distance as floating point usually for attention bias? 
        # Actually BrepEncoder `batch_data["d2_distance"]` is passed to GraphAttnBias.
        # Inside GraphAttnBias, it likely uses it. Let's assume float32 to be safe for Histograms.
        # But wait, logic was bins.
        # Let's check collator: `pad_d2_pos_unsqueeze` -> dtype=x.dtype.
        # GraphAttnBias usually embeds them? 
        # If it's a histogram, it might be used as features.
        # I will keep as float32 for safety as histograms are counts/densities.
        
        "angle_distance": torch.tensor(data['angle_distance'], dtype=torch.float32)
    }
    
    return g, aux

if __name__ == "__main__":
    pass
