# gnn_batch_builder.py
import os
import torch
from parse_kicad_gnn import parse_kicad
from graph_builder0 import build_net_graph

def build_graph_dataset(input_dir="../data", output_dir="../graphs", edge_dist_thresh=60.0):
    """
    Iterates over all .kicad_pcb files in input_dir,
    builds enriched net-level graphs (from graph_builder.py),
    and saves each as a .pt PyTorch Geometric Data object.
    """
    os.makedirs(output_dir, exist_ok=True)
    pcb_files = [f for f in os.listdir(input_dir) if f.endswith(".kicad_pcb")]

    if not pcb_files:
        print(f"No .kicad_pcb files found in {input_dir}")
        return []

    graphs = []

    for fname in pcb_files:
        fpath = os.path.join(input_dir, fname)
        try:
            # --- Step 1: Parse KiCad PCB file
            meta = parse_kicad(fpath)
            if not meta or "net_pads" not in meta:
                print(f" Skipped {fname}: invalid parse output")
                continue

            # --- Step 2: Build net-level graph with enriched features
            graph = build_net_graph(meta, edge_dist_thresh=edge_dist_thresh)

            # --- Step 3: Attach metadata for traceability
            graph.board_name = fname.replace(".kicad_pcb", "")
            graph.num_nodes = len(graph.x)
            graph.num_edges = graph.edge_index.size(1) // 2
            graph.num_node_features = graph.x.size(1)
            graph.num_edge_features = graph.edge_attr.size(1)

            # --- Step 4: Save .pt file
            out_path = os.path.join(output_dir, fname.replace(".kicad_pcb", ".pt"))
            torch.save(graph, out_path)

            graphs.append(graph)
            print(f" Built {fname}: {graph.num_nodes} nets, {graph.num_edges} edges "
                  f"({graph.num_node_features} node feats, {graph.num_edge_features} edge feats)")

        except Exception as e:
            print(f" Error processing {fname}: {e}")

    print(f"\n Saved {len(graphs)} graphs to {output_dir}")
    return graphs


if __name__ == "__main__":
    build_graph_dataset()
