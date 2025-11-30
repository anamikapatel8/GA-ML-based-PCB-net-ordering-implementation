import numpy as np

def board_features(meta, grid=None):
    """Compute geometric and density features of the board."""
    bbox = meta["board_bbox"]
    if not bbox:
        return {}
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    area = w * h
    pin_density = meta["pad_count"] / area if area > 0 else 0
    aspect_ratio = w / h if h != 0 else 0
    obstacle_density = np.mean(grid == 1) if grid is not None else 0.0

    return {
        "width": w,
        "height": h,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "pin_density": pin_density,
        "obstacle_density": obstacle_density
    }


def compute_net_geometry(nets):
    """Compute per-net geometric features: Manhattan distance, bbox area, pin count."""
    net_feats = []
    for net, pins in nets.items():
        if len(pins) < 2:
            continue
        xs, ys = zip(*pins)
        bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        # Mean pairwise Manhattan length
        dists = [
            abs(xs[i]-xs[j]) + abs(ys[i]-ys[j])
            for i in range(len(xs)) for j in range(i+1, len(xs))
        ]
        manhattan = np.mean(dists)
        net_feats.append({
            "net": net,
            "pin_count": len(pins),
            "manhattan": manhattan,
            "bbox_area": bbox_area
        })
    return net_feats


def aggregate_net_features(net_feats):
    """Aggregate net-level geometry into summary stats."""
    if not net_feats:
        return {}
    pins = np.array([n["pin_count"] for n in net_feats])
    manh = np.array([n["manhattan"] for n in net_feats])
    bbox = np.array([n["bbox_area"] for n in net_feats])

    return {
        "mean_net_pins": np.mean(pins),
        "std_net_pins": np.std(pins),
        "mean_manhattan_len": np.mean(manh),
        "std_manhattan_len": np.std(manh),
        "frac_long_nets": np.mean(manh > np.median(manh)),
        "avg_net_bbox_area": np.mean(bbox)
    }


def routing_metrics(net_results, total_nets, total_wirelength, success_rate, grid):
    """Aggregate routing-level features."""
    if not net_results:
        return {}
    routed = [r for r in net_results if r["success"]]
    mean_routed_len = np.mean([r["routed_len"] for r in routed]) if routed else 0
    mean_stretch = np.mean([r["stretch"] for r in routed]) if routed else 0
    final_occupied = np.mean(grid == 1)
    routing_congestion = np.var(grid)  # variance in occupancy = spatial congestion
    avg_failed_len_est = np.mean([r["manhattan"] for r in net_results if not r["success"]]) \
                     if any(not r["success"] for r in net_results) else 0.0

    routing_eff = success_rate / (mean_stretch + 1e-6) if mean_stretch > 0 else 0

    return {
        "mean_routed_len": mean_routed_len,
        "mean_stretch": mean_stretch,
        "routing_efficiency": routing_eff,
        "final_occupied_ratio": final_occupied,
        "routing_congestion": routing_congestion,
        "avg_failed_len_est": avg_failed_len_est
    }

# ------------------- Net ordering–based features -------------------

def ordering_features(meta, order, coarse_res=2.0, alpha=0.4, k_neighbor=5):
    """
    Simulate routing given an ordering of nets and extract congestion-aware features.
    Returns (agg_features, per_net_features)
    """
    bbox = meta["board_bbox"]
    if not bbox:
        return {}, []

    W = int((bbox[2]-bbox[0]) / coarse_res) + 1
    H = int((bbox[3]-bbox[1]) / coarse_res) + 1
    occ = np.zeros((H, W), dtype=np.uint8)

    # --- precompute per-net geometry ---
    def manhattan(p1, p2): return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    def manhattan_net(pins):
        if len(pins) == 2:
            return manhattan(pins[0], pins[1])
        pts = list(pins)
        visited = [pts.pop(0)]
        total = 0
        while pts:
            dmin, idx = float("inf"), None
            for i, p in enumerate(pts):
                for q in visited:
                    d = manhattan(p, q)
                    if d < dmin:
                        dmin, idx = d, i
            total += dmin
            visited.append(pts.pop(idx))
        return total

    def bbox_of_net(pins):
        xs, ys = zip(*pins)
        return min(xs), min(ys), max(xs), max(ys)

    net_geom = {}
    for net, pins in meta["nets"].items():
        if len(pins) < 2:
            continue
        mlen = manhattan_net(pins)
        bx0, by0, bx1, by1 = bbox_of_net(pins)
        net_geom[net] = {
            "manhattan": mlen,
            "bbox": (bx0, by0, bx1, by1),
            "pin_count": len(pins)
        }

    per_net = []
    total_est_len = 0.0

    def mark_bbox(b):
        x0, y0, x1, y1 = b
        gx0 = max(0, int((x0 - bbox[0]) / coarse_res))
        gy0 = max(0, int((y0 - bbox[1]) / coarse_res))
        gx1 = min(W - 1, int((x1 - bbox[0]) / coarse_res))
        gy1 = min(H - 1, int((y1 - bbox[1]) / coarse_res))
        cells, overlap = 0, 0
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                cells += 1
                if occ[gy, gx]:
                    overlap += 1
                else:
                    occ[gy, gx] = 1
        return cells, overlap

    # --- simulate routing according to ordering ---
    for pos, net in enumerate(order):
        if net not in net_geom:
            continue
        g = net_geom[net]
        mlen = g["manhattan"]
        bbox_n = g["bbox"]
        cells, overlap = mark_bbox(bbox_n)
        overlap_frac = overlap / cells if cells > 0 else 0.0
        est_len = mlen * (1 + alpha * overlap_frac)
        total_est_len += est_len
        per_net.append({
            "net": net,
            "pos_norm": pos / max(1, len(order)-1),
            "manhattan_len": mlen,
            "bbox_area": (bbox_n[2]-bbox_n[0])*(bbox_n[3]-bbox_n[1]),
            "overlap_frac": overlap_frac,
            "est_routed_len": est_len,
        })

    # --- aggregate features ---
    if not per_net:
        return {}, []

    man_arr = np.array([p["manhattan_len"] for p in per_net])
    overlap_arr = np.array([p["overlap_frac"] for p in per_net])
    pos_arr = np.array([p["pos_norm"] for p in per_net])
    long_mask = man_arr > np.percentile(man_arr, 75)

    agg = {
        "total_est_len": float(total_est_len),
        "mean_manhattan": float(man_arr.mean()),
        "std_manhattan": float(man_arr.std()),
        "mean_overlap": float(overlap_arr.mean()),
        "std_overlap": float(overlap_arr.std()),
        "frac_long_nets_early": float(np.mean(long_mask[pos_arr < 0.25])) if len(man_arr) > 0 else 0,
        "final_occupied_ratio": float(occ.mean())
    }

    return agg, per_net
