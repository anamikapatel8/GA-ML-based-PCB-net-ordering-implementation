import torch
import numpy as np
from torch_geometric.data import Data
from parse_kicad import parse_kicad


def build_net_graph(meta, edge_dist_thresh=60.0):
    """
    Builds an enriched net-level graph for GNN learning.

    Nodes = nets
    Edges = (a) shared components (electrical)
             (b) spatial proximity (geometric)

    Node features (now 12 features total):
    [
        0 pin_count,
        1 bbox_area,
        2 mean_manhattan,
        3 pin_density_norm,
        4 bbox_aspect_ratio,
        5 relative_area_frac,
        6 mean_inter_net_dist,
        7 component_count,
        8 pos_norm,
        9 overlap_frac,
        10 avg_component_area,      <-- NEW: avg total component area connected to net
        11 local_congestion_density <-- NEW: how crowded is the region around this net
    ]

    Edge features = [
        shared_comp_count,
        centroid_distance,
        bbox_overlap_ratio,
        net_size_ratio,
        edge_type (1=shared component, 0=proximity)
    ]
    """

    nets = meta["net_pads"]
    net_names = list(nets.keys())
    N = len(net_names)

    board_bbox = meta.get("board_bbox", [0, 0, 100, 100])
    board_w = board_bbox[2] - board_bbox[0]
    board_h = board_bbox[3] - board_bbox[1]
    board_area = board_w * board_h if board_w > 0 and board_h > 0 else 1.0

    # Component areas if provided (from parse_kicad)
    comp_areas = meta.get("comp_areas", {})

    node_feats = []
    centroids, bboxes = {}, {}

    # -------------------------------
    # Pass 1: Compute basic net geometry and features
    # -------------------------------
    for net_name, pads in nets.items():
        if not pads:
            node_feats.append([0] * 12)
            centroids[net_name] = (0, 0)
            bboxes[net_name] = (0, 0, 0, 0)
            continue

        xs = [p["x"] for p in pads]
        ys = [p["y"] for p in pads]
        pin_count = len(xs)

        bbox_area = max(1e-3, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        aspect_ratio = (max(xs) - min(xs)) / max(1e-3, (max(ys) - min(ys)))
        relative_area_frac = bbox_area / board_area
        comp_names = {p["comp"] for p in pads}
        comp_count = len(comp_names)

        # Mean internal Manhattan distance
        manhattan = 0.0
        if pin_count > 1:
            dsum, cnt = 0.0, 0
            for i in range(pin_count):
                for j in range(i + 1, pin_count):
                    dsum += abs(xs[i] - xs[j]) + abs(ys[i] - ys[j])
                    cnt += 1
            manhattan = dsum / cnt if cnt > 0 else 0.0

        pin_density = pin_count / bbox_area

        # --- NEW: average connected component area
        if comp_areas:
            comp_area_vals = [comp_areas.get(c, 0.0) for c in comp_names]
            avg_comp_area = np.mean(comp_area_vals) if comp_area_vals else 0.0
        else:
            avg_comp_area = 0.0

        centroids[net_name] = (np.mean(xs), np.mean(ys))
        bboxes[net_name] = (min(xs), min(ys), max(xs), max(ys))

        # temporary placeholders for inter-net dist, pos_norm, overlap_frac, congestion
        node_feats.append([
            pin_count, bbox_area, manhattan,
            pin_density, aspect_ratio, relative_area_frac,
            0.0, comp_count, 0.0, 0.0, avg_comp_area, 0.0
        ])

    # -------------------------------
    # Helper: compute bbox overlap
    # -------------------------------
    def bbox_overlap(b1, b2):
        x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        overlap_area = x_overlap * y_overlap
        union_area = (
            (b1[2] - b1[0]) * (b1[3] - b1[1])
            + (b2[2] - b2[0]) * (b2[3] - b2[1])
            - overlap_area
        )
        return overlap_area / union_area if union_area > 0 else 0.0

    # -------------------------------
    # Pass 2: compute inter-net relationships and local congestion
    # -------------------------------
    avg_bbox_size = np.mean([bboxes[n][2] - bboxes[n][0] for n in net_names]) if N > 0 else 1.0

    for i, ni in enumerate(net_names):
        ci = np.array(centroids[ni])
        dists, overlaps = [], []

        # For congestion measure
        local_neighbors = 0

        for j, nj in enumerate(net_names):
            if i == j:
                continue
            cj = np.array(centroids[nj])
            dist = np.linalg.norm(ci - cj)
            dists.append(dist)
            overlaps.append(bbox_overlap(bboxes[ni], bboxes[nj]))

            # Count neighbors within local area (~2× avg bbox size)
            if dist < 2.0 * avg_bbox_size:
                local_neighbors += 1

        if dists:
            node_feats[i][6] = np.mean(dists)  # mean_inter_net_dist
        node_feats[i][8] = np.mean([ci[0] / board_w, ci[1] / board_h])  # pos_norm
        node_feats[i][9] = np.mean(overlaps) if overlaps else 0.0         # overlap_frac

        # --- NEW: local congestion density ---
        node_feats[i][11] = local_neighbors / max(1, N - 1)

    # -------------------------------
    # Pass 3: Build edges (shared components or proximity)
    # -------------------------------
    edge_list, edge_attrs = [], []
    for i in range(N):
        ni = net_names[i]
        for j in range(i + 1, N):
            nj = net_names[j]

            comps_i = {p["comp"] for p in nets.get(ni, [])}
            comps_j = {p["comp"] for p in nets.get(nj, [])}
            shared_count = len(comps_i.intersection(comps_j))

            ci, cj = np.array(centroids[ni]), np.array(centroids[nj])
            dist = np.linalg.norm(ci - cj)
            overlap = bbox_overlap(bboxes[ni], bboxes[nj])
            area_ratio = min(bboxes[ni][2] - bboxes[ni][0], bboxes[nj][2] - bboxes[nj][0]) / \
                         max(bboxes[ni][2] - bboxes[ni][0], bboxes[nj][2] - bboxes[nj][0] + 1e-6)
            edge_type = 1.0 if shared_count > 0 else 0.0

            if shared_count > 0 or dist <= edge_dist_thresh:
                edge_list += [[i, j], [j, i]]
                edge_attrs += [[shared_count, dist, overlap, area_ratio, edge_type]] * 2

    # Convert to torch tensors
    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = (
        torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        if edge_list
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_attrs, dtype=torch.float)
        if edge_attrs
        else torch.empty((0, 5), dtype=torch.float)
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.net_names = net_names
    data.board_bbox = board_bbox
    return data


if __name__ == "__main__":
    meta = parse_kicad("../data/d1_mini_shield.kicad_pcb")
    g = build_net_graph(meta)
    print(f"Graph built: {len(g.x)} nets, {g.edge_index.size(1)//2} edges")
    print("Example node features (first 5):", g.x[:5])
    print("Edge attrs shape:", g.edge_attr.shape)
