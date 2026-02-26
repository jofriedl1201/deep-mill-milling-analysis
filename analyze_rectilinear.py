"""Detailed rectilinear parts analysis"""
import sys
sys.path.insert(0, 'BREP Graph Generator')
from analysis_engine.brep_graph import BRepGraphGenerator
from analysis_engine.brep_mfr import GraphNativeFeatureRecognizer

parts = [
    ('DOOR MOTOR MOUNT', 'Step Files/S-U09900012-000__MOUNT - TADS V2.0 DOOR MOTOR MOUNT.step'),
    ('Left side Traitener', 'Step Files/Left side Traitener.step')
]

gen = BRepGraphGenerator(use_fast_path=True)
rec = GraphNativeFeatureRecognizer()

for name, path in parts:
    print(f"\n{'='*60}")
    print(f"PART: {name}")
    print('='*60)
    
    g = gen.generate(path)
    d = rec.analyze(g)
    
    print(f"\nGraph Statistics:")
    print(f"  Faces: {g.num_faces}")
    print(f"  Edges: {g.num_edges}")
    print(f"  Hash: {g.graph_hash}")
    
    # Surface type distribution
    type_names = {0: "Plane", 1: "Cylinder", 2: "Cone", 3: "Sphere", 4: "Torus", 5: "Bezier", 6: "BSpline", 7: "Other"}
    type_counts = {}
    for f in g.faces.values():
        st = f.get("surface_type", 7)
        type_counts[st] = type_counts.get(st, 0) + 1
    
    print(f"\nSurface Types:")
    for st, cnt in sorted(type_counts.items()):
        print(f"  {type_names.get(st, 'Unknown')}: {cnt}")
    
    # Diagnostic labels
    label_counts = {}
    for diag in d.values():
        label_counts[diag.diagnostic_label] = label_counts.get(diag.diagnostic_label, 0) + 1
    
    print(f"\nDiagnostic Labels:")
    for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / g.num_faces
        print(f"  {label}: {cnt} ({pct:.1f}%)")
    
    # Confidence distribution
    conf_counts = {'high': 0, 'medium': 0, 'low': 0}
    for diag in d.values():
        conf_counts[diag.confidence] += 1
    
    total = sum(conf_counts.values())
    print(f"\nConfidence Distribution:")
    for conf in ['high', 'medium', 'low']:
        cnt = conf_counts[conf]
        pct = 100 * cnt / total if total > 0 else 0
        print(f"  {conf.upper()}: {cnt} ({pct:.1f}%)")

print("\n" + "="*60)
print("COMPARISON TO FULL DATASET (5 parts, 2339 faces)")
print("="*60)
print("""
Full Dataset Confidence:
  HIGH: 459 (19.6%)
  MEDIUM: 387 (16.5%)
  LOW: 1493 (63.8%)

Rectilinear Parts Confidence (see above) should show:
- Higher proportion of HIGH confidence (simpler geometry)
- More consistent label mapping (fewer surface types)
""")
