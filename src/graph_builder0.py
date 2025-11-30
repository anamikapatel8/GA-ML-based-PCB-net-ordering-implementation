import torch
import numpy as np
from torch_geometric.data import Data
from parse_kicad_gnn import parse_kicad

def build_net_graph(meta, edge_dist_thresh=60.0):
    """
    Builds an enriched net-level graph for GNN learning.

    Nodes = nets
    Edges = (a) shared components (electrical)
             (b) spatial proximity (geometric)
    Node features = [
        pin_count, bbox_area, mean_manhattan,
        pin_density_norm, bbox_aspect_ratio,
        relative_area_frac, mean_inter_net_dist,
        component_count, pos_norm, overlap_frac
    ]
    Edge features = [shared_comp_count, centroid_distance,
                     bbox_overlap_ratio, net_size_ratio, edge_type]
    """
    nets = meta["net_pads"]
    net_names = list(nets.keys())
    N = len(net_names)
    board_bbox = meta.get("board_bbox", [0, 0, 100, 100])
    board_w = board_bbox[2] - board_bbox[0]
    board_h = board_bbox[3] - board_bbox[1]
    board_area = board_w * board_h if board_w > 0 and board_h > 0 else 1.0

    node_feats = []
    centroids, bboxes = {}, {}

    for net_name, pads in nets.items():
        if not pads:
            node_feats.append([0]*10)
            centroids[net_name] = (0, 0)
            bboxes[net_name] = (0, 0, 0, 0)
            continue

        xs = [p["x"] for p in pads]
        ys = [p["y"] for p in pads]
        pin_count = len(xs)
        bbox_area = max(1e-3, (max(xs)-min(xs)) * (max(ys)-min(ys)))
        aspect_ratio = (max(xs)-min(xs)) / max(1e-3, (max(ys)-min(ys)))
        relative_area_frac = bbox_area / board_area

        comp_count = len({p["comp"] for p in pads})

        # Mean internal Manhattan distance
        manhattan = 0.0
        if pin_count > 1:
            dsum, cnt = 0.0, 0
            for i in range(pin_count):
                for j in range(i + 1, pin_count):
                    dsum += abs(xs[i]-xs[j]) + abs(ys[i]-ys[j])
                    cnt += 1
            manhattan = dsum / cnt if cnt > 0 else 0.0

        pin_density = pin_count / bbox_area

        centroids[net_name] = (np.mean(xs), np.mean(ys))
        bboxes[net_name] = (min(xs), min(ys), max(xs), max(ys))

        # Temporary placeholders for mean_inter_net_dist, pos_norm, overlap_frac
        node_feats.append([
            pin_count, bbox_area, manhattan,
            pin_density, aspect_ratio, relative_area_frac,
            0.0, comp_count, 0.0, 0.0
        ])

    # --- Mean distance, pos_norm, overlap_frac ---
    def bbox_overlap(b1, b2):
        x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        overlap_area = x_overlap * y_overlap
        union_area = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - overlap_area
        return overlap_area / union_area if union_area > 0 else 0.0

    for i, ni in enumerate(net_names):
        ci = np.array(centroids[ni])
        dists = []
        overlaps = []
        for j, nj in enumerate(net_names):
            if i == j:
                continue
            cj = np.array(centroids[nj])
            dists.append(np.linalg.norm(ci - cj))
            overlaps.append(bbox_overlap(bboxes[ni], bboxes[nj]))

        if dists:
            node_feats[i][6] = np.mean(dists)
        node_feats[i][8] = np.mean([ci[0] / board_w, ci[1] / board_h])  # pos_norm
        node_feats[i][9] = np.mean(overlaps) if overlaps else 0.0         # overlap_frac

    # --- Edges ---
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
            area_ratio = min(bboxes[ni][2]-bboxes[ni][0], bboxes[nj][2]-bboxes[nj][0]) / \
                         max(bboxes[ni][2]-bboxes[ni][0], bboxes[nj][2]-bboxes[nj][0] + 1e-6)
            edge_type = 1.0 if shared_count > 0 else 0.0

            if shared_count > 0 or dist <= edge_dist_thresh:
                edge_list += [[i, j], [j, i]]
                edge_attrs += [[shared_count, dist, overlap, area_ratio, edge_type]] * 2

    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.empty((0, 5), dtype=torch.float)

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
