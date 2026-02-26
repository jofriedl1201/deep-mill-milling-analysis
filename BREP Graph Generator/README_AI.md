# BrepMFR Graph Generator & Inference Pipeline

> [!IMPORTANT]
> **BrepMFR Model Status: RESEARCH REFERENCE ONLY**
> 
> The original BrepMFR model is NOT currently executable due to:
> - Missing preprocessing code in the public repository
> - No pretrained checkpoints available for download
> - DGL version incompatibility with current environment (DGL 2.2.1 requires PyTorch ≤2.4)
> 
> This generator produces **BrepMFR-compatible encodings** that can be consumed by:
> - The `GraphNativeFeatureRecognizer` (rule-based diagnostics)
> - Future ML models when checkpoints become available
> - Custom GNN implementations
> 
> See [GRAPH_SCHEMA.md](./GRAPH_SCHEMA.md) for the canonical graph structure documentation.

## Overview
This pipeline converts a STEP file (CAD B-Rep) into a graph representation for machining feature analysis. The graph structure is compatible with the [BrepMFR](https://github.com/zhangshuming0668/BrepMFR) research model.

## Directory Structure
- `brep_step1.py`: **Steps 1-3**. Loads STEP files, extracts face/edge geometry, samples UV grids, and computes global relation tensors (`spatial_pos`, `d2_distance`, `angle_distance`).
- `brep_dgl.py`: **Step 4**. Assembles the extraction results into a bidirectional `dgl.DGLGraph` and a dictionary of auxiliary global tensors.
- `brep_inference.py`: **Step 5**. Orchestrator. Runs the extraction, wraps data into a `PYGGraph`-like object, batches it using the official `collator`, and runs inference using a text checkpoint.

## Dependencies
- **Python Libraries**: `pythonocc-core`, `dgl`, `torch`, `numpy`
- **External Repo**: A clone of **BrepMFR** must be present in `../BrepMFR` or `./BrepMFR`.

## Usage
The entire pipeline is driven by `brep_inference.py`.

```bash
python brep_inference.py --step_file <path_to_step_file> --checkpoint <path_to_model_checkpoint>
```

### Arguments
- `--step_file`: Absolute path to the input `.step` or `.stp` file.
- `--checkpoint`: Absolute path to the BrepMFR model checkpoint (`.ckpt`).

### output
Prints a list of Face IDs and their predicted feature class indices.
```text
Running Step 1-3 Analysis...
Running Step 4 DGL Assembly...
Running Step 5 Inference...
Predictions:
Face 0: 4
Face 1: 12
...
```

## Implementation Details for AI Agents

### 1. Feature Extraction (`brep_step1.py`)
- **Faces**: Uses `BRepAdaptor_Surface` for properties. Deterministic ID based on `HashCode(MAX_INT)`.
- **Edges**: Uses `BRepAdaptor_Curve` for 5x6 grid. Computes Dihedral Angle and Convexity analytically.
- **Global Tensors**:
    - `spatial_pos` (A1): APSP (All-Pairs Shortest Path) on face adjacency graph (BFS).
    - `d2_distance`: Pairwise D2 (Shape Distribution) histograms (64 bins). Sampled from Union(Face A, Face B).
    - `angle_distance`: Global A3 (Shape Distribution) histogram (64 bins). Replicated/Broadcasted.
    - `edge_path`: Indices of edges along shortest paths. Padded with -1. Max length 16.

### 2. Graph Assembly (`brep_dgl.py`)
- Creates a **Bidirectional** Graph.
- Node Features: `x` (Grid), `z` (Type), `y` (Area), `l` (Loop), `a` (Adj), `f` (Label).
- Edge Features: `x` (Grid), `t` (Type), `l` (Len), `a` (Ang), `c` (Conv).
- **Strictly** maps types: `float32` for features/histograms, `int64` (long) for indices/types.

### 3. Inference (`brep_inference.py`)
- Dynamically imports `BrepSeg` and `collator` from the local `BrepMFR` repo.
- Adapts DGL graph -> `PYGGraph` object to mimic `dataset.py` loading.
- Uses `collator` to handle padding and batching (Batch Size = 1).
- Runs `model.brep_encoder` and `model.classifier`.
- **Note**: Does not use `trainer.test()` to avoid file I/O overhead. Manually executes forward pass logic.
