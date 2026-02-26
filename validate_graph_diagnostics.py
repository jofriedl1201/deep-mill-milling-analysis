"""
Graph Diagnostic Validation Script

Runs graph generation and feature diagnostics across all STEP files,
collects statistics, and produces a confidence report.
"""

import os
import sys
import json
from collections import defaultdict

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BREP Graph Generator"))
sys.path.insert(0, os.path.dirname(__file__))

from analysis_engine.brep_graph import BRepGraphGenerator, extract_subgraph
from analysis_engine.brep_mfr import GraphNativeFeatureRecognizer

def analyze_part(step_path: str, generator: BRepGraphGenerator, recognizer: GraphNativeFeatureRecognizer):
    """Analyze a single part and return statistics."""
    try:
        graph = generator.generate(step_path)
        diagnostics = recognizer.analyze(graph)
        
        # Count surface types
        type_counts = defaultdict(int)
        for fdata in graph.faces.values():
            st = fdata.get("surface_type", 7)
            type_counts[st] += 1
        
        # Count diagnostic labels
        label_counts = defaultdict(int)
        for diag in diagnostics.values():
            label_counts[diag.diagnostic_label] += 1
        
        # Confidence distribution
        confidence_counts = defaultdict(int)
        for diag in diagnostics.values():
            confidence_counts[diag.confidence] += 1
        
        # Edge convexity stats
        convex_count = len([e for e in graph.edges.values() if e.get("convexity") == 1])
        concave_count = len([e for e in graph.edges.values() if e.get("convexity") == 0])
        tangent_count = len([e for e in graph.edges.values() if e.get("convexity") == 2])
        
        return {
            "success": True,
            "file": os.path.basename(step_path),
            "num_faces": graph.num_faces,
            "num_edges": graph.num_edges,
            "graph_hash": graph.graph_hash,
            "surface_types": dict(type_counts),
            "edge_convexity": {
                "convex": convex_count,
                "concave": concave_count,
                "tangent": tangent_count
            },
            "diagnostic_labels": dict(label_counts),
            "confidence_dist": dict(confidence_counts),
            "diagnostics_detail": {fid: {
                "surface_type": d.surface_type,
                "label": d.diagnostic_label,
                "confidence": d.confidence,
                "adj_count": d.adjacent_count,
                "convex_edges": d.convex_edge_count,
                "concave_edges": d.concave_edge_count
            } for fid, d in diagnostics.items()}
        }
    except Exception as e:
        return {
            "success": False,
            "file": os.path.basename(step_path),
            "error": str(e)
        }

def main():
    step_dir = "Step Files"
    step_files = [f for f in os.listdir(step_dir) if f.lower().endswith(('.step', '.stp'))]
    
    print(f"=== Graph Diagnostic Validation ===")
    print(f"Found {len(step_files)} STEP files\n")
    
    generator = BRepGraphGenerator(use_fast_path=True, time_budget_sec=10.0)
    recognizer = GraphNativeFeatureRecognizer()
    
    results = []
    
    for step_file in step_files:
        step_path = os.path.join(step_dir, step_file)
        print(f"Processing: {step_file}...")
        
        result = analyze_part(step_path, generator, recognizer)
        results.append(result)
        
        if result["success"]:
            print(f"  Faces: {result['num_faces']}, Edges: {result['num_edges']}, Hash: {result['graph_hash'][:8]}")
        else:
            print(f"  ERROR: {result['error']}")
    
    # ============================================
    # Aggregate Analysis
    # ============================================
    print("\n" + "="*60)
    print("AGGREGATE ANALYSIS")
    print("="*60)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\nSuccess Rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
    
    if failed:
        print(f"\nFailed Parts:")
        for f in failed:
            print(f"  - {f['file']}: {f['error']}")
    
    # Surface Type Distribution
    print("\n--- Surface Type Distribution (All Parts) ---")
    type_names = {0: "Plane", 1: "Cylinder", 2: "Cone", 3: "Sphere", 
                  4: "Torus", 5: "Bezier", 6: "BSpline", 7: "Other"}
    total_type_counts = defaultdict(int)
    for r in successful:
        for st, cnt in r["surface_types"].items():
            total_type_counts[int(st)] += cnt
    
    total_faces = sum(total_type_counts.values())
    for st in sorted(total_type_counts.keys()):
        name = type_names.get(st, f"Type{st}")
        cnt = total_type_counts[st]
        pct = 100 * cnt / total_faces if total_faces > 0 else 0
        print(f"  {name}: {cnt} ({pct:.1f}%)")
    
    # Diagnostic Label Distribution
    print("\n--- Diagnostic Label Distribution (All Parts) ---")
    total_label_counts = defaultdict(int)
    for r in successful:
        for label, cnt in r["diagnostic_labels"].items():
            total_label_counts[label] += cnt
    
    for label in sorted(total_label_counts.keys(), key=lambda x: -total_label_counts[x]):
        cnt = total_label_counts[label]
        pct = 100 * cnt / total_faces if total_faces > 0 else 0
        print(f"  {label}: {cnt} ({pct:.1f}%)")
    
    # Confidence Distribution
    print("\n--- Confidence Distribution ---")
    total_conf = defaultdict(int)
    for r in successful:
        for conf, cnt in r["confidence_dist"].items():
            total_conf[conf] += cnt
    
    for conf in ["high", "medium", "low"]:
        cnt = total_conf.get(conf, 0)
        pct = 100 * cnt / total_faces if total_faces > 0 else 0
        print(f"  {conf.upper()}: {cnt} ({pct:.1f}%)")
    
    # Per-Part Statistics
    print("\n--- Per-Part Statistics ---")
    print(f"{'Part':<50} {'Faces':>6} {'Edges':>6} {'Labels':>8}")
    print("-"*75)
    for r in successful:
        name = r["file"][:47] + "..." if len(r["file"]) > 50 else r["file"]
        print(f"{name:<50} {r['num_faces']:>6} {r['num_edges']:>6} {len(r['diagnostic_labels']):>8}")
    
    # Consistency Analysis
    print("\n--- Diagnostic Consistency Analysis ---")
    
    # Check if same surface type gets consistent labels
    type_to_labels = defaultdict(lambda: defaultdict(int))
    for r in successful:
        for fid, detail in r["diagnostics_detail"].items():
            stype = detail["surface_type"]
            label = detail["label"]
            type_to_labels[stype][label] += 1
    
    print("\nSurface Type -> Label Mapping Consistency:")
    for stype in sorted(type_to_labels.keys()):
        labels = type_to_labels[stype]
        total = sum(labels.values())
        top_label = max(labels.keys(), key=lambda x: labels[x])
        consistency = labels[top_label] / total if total > 0 else 0
        
        print(f"\n  {stype} ({total} faces):")
        for label, cnt in sorted(labels.items(), key=lambda x: -x[1])[:3]:
            pct = 100 * cnt / total if total > 0 else 0
            print(f"    -> {label}: {cnt} ({pct:.1f}%)")
        print(f"    Consistency: {100*consistency:.1f}% (top label)")
    
    # Edge Convexity Impact
    print("\n--- Edge Convexity Impact on Diagnostics ---")
    high_concave_faces = []
    for r in successful:
        for fid, detail in r["diagnostics_detail"].items():
            if detail["concave_edges"] >= 3:
                high_concave_faces.append({
                    "part": r["file"],
                    "face_id": fid,
                    **detail
                })
    
    if high_concave_faces:
        print(f"Faces with 3+ concave edges: {len(high_concave_faces)}")
        label_for_concave = defaultdict(int)
        for f in high_concave_faces:
            label_for_concave[f["label"]] += 1
        for label, cnt in sorted(label_for_concave.items(), key=lambda x: -x[1]):
            print(f"  {label}: {cnt}")
    else:
        print("No faces with 3+ concave edges found.")
    
    # Weakness Identification
    print("\n" + "="*60)
    print("WEAKNESS IDENTIFICATION")
    print("="*60)
    
    low_conf_pct = 100 * total_conf.get("low", 0) / total_faces if total_faces > 0 else 0
    print(f"\n1. Low Confidence Rate: {low_conf_pct:.1f}%")
    if low_conf_pct > 50:
        print("   WARNING: Majority of diagnostics have low confidence.")
    
    # Find labels with high variance
    print("\n2. Labels with Inconsistent Confidence:")
    for r in successful:
        label_conf = defaultdict(list)
        for fid, detail in r["diagnostics_detail"].items():
            label_conf[detail["label"]].append(detail["confidence"])
        
        for label, confs in label_conf.items():
            unique_confs = set(confs)
            if len(unique_confs) > 1:
                print(f"   {r['file'][:30]}: '{label}' has mixed confidence ({', '.join(unique_confs)})")
    
    print("\n3. Heuristic Gaps:")
    if total_type_counts.get(7, 0) > 0:
        print(f"   - 'Other' surface type: {total_type_counts[7]} faces (not fully classified)")
    
    cylindrical_labels = type_to_labels.get("Cylinder", {})
    if len(cylindrical_labels) > 2:
        print(f"   - Cylindrical surfaces map to {len(cylindrical_labels)} different labels (high variability)")
    
    print("\n=== Validation Complete ===")

if __name__ == "__main__":
    main()
