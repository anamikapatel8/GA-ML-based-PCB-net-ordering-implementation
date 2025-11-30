import json
import numpy as np
from parse_kicad_gnn import parse_kicad
from astar import astar
from feature_extractor import board_features, net_features

# === Parse board ===
meta = parse_kicad("../data/d1_mini_shield.kicad_pcb")
nets = meta["nets"]
print("Nets found:", list(nets.keys()))

# === Build coarse grid ===
bbox = meta["board_bbox"]
scale = 1  # 1mm per cell (can refine)
W, H = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
grid = np.zeros((H, W), dtype=int)

# mark edges as blocked
grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1

# === Route all nets sequentially ===
total_length = 0
successful = 0
failed_nets = []

for net_name, pins in nets.items():
    if len(pins) < 2:
        continue

    # route pairwise for now (first two pins only)
    p1, p2 = pins[0], pins[1]
    s = (int(p1[1] - bbox[1]), int(p1[0] - bbox[0]))  # (row, col)
    g = (int(p2[1] - bbox[1]), int(p2[0] - bbox[0]))

    print(f"Routing net {net_name}: {p1} to {p2}")
    path = astar(s, g, grid)

    if path:
        successful += 1
        total_length += len(path)

        # mark path as occupied
        for (y, x) in path:
            if 0 <= y < H and 0 <= x < W:
                grid[y, x] = 1
        print(f"  Routed ({len(path)} cells)")
    else:
        failed_nets.append(net_name)
        print("  Routing failed")

# === Compute summary metrics ===
total_nets = len([n for n in nets if len(nets[n]) >= 2])
success_rate = (successful / total_nets * 100) if total_nets > 0 else 0
via_count = 0  # placeholder for future multilayer routing

print("\n=== Routing Summary ===")
print(f"Total nets considered : {total_nets}")
print(f"Successfully routed   : {successful}")
print(f"Failed nets           : {failed_nets if failed_nets else 'None'}")
print(f"Total wirelength (cells): {total_length}")
print(f"Success rate          : {success_rate:.2f}%")
print(f"Via count             : {via_count}")

# === Features ===
print("\n=== Board & Net Features ===")
print("Board features:", board_features(meta))
print("Net features:", net_features(meta))
