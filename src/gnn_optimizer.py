# gnn_optimizer.py
import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from graph_builder0 import build_net_graph
from parse_kicad_gnn import parse_kicad
import random
import copy
import matplotlib.pyplot as plt

# ===============================================================
# Model Definition (same as train_gnn.py)
# ===============================================================
class NetCostGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        out = self.lin(h)
        return out.squeeze(), h


# ===============================================================
#  Genetic Algorithm for Net Ordering
# ===============================================================
def ga_optimize(scores, pop_size=30, n_gen=40, elite_frac=0.2, mutation_rate=0.2):
    """
    Optimize net ordering using GA over GNN-predicted scores.
    Lower total score = better.
    """
    num_nets = len(scores)
    if num_nets <= 2:
        return list(range(num_nets)), float(np.sum(scores))

    nets = list(range(num_nets))
    population = [random.sample(nets, num_nets) for _ in range(pop_size)]

    def fitness(order):
        # Penalize early placement of high-cost nets
        cost = sum((pos / num_nets) * scores[i] for pos, i in enumerate(order))
        return cost

    for _ in range(n_gen):
        fitness_vals = np.array([fitness(p) for p in population])
        sorted_idx = np.argsort(fitness_vals)
        elites = [population[i] for i in sorted_idx[:max(1, int(pop_size * elite_frac))]]

        new_pop = copy.deepcopy(elites)
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            cut = random.randint(1, num_nets - 2)
            child = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
            if random.random() < mutation_rate:
                i, j = random.sample(range(num_nets), 2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        population = new_pop

    best = min(population, key=fitness)
    return best, fitness(best)


# ===============================================================
#  Visualization Helper
# ===============================================================
def plot_net_ordering(board, net_names, scores, gnn_order, ga_order, out_dir="../results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))

    gnn_costs = [scores[i] for i in gnn_order]
    ga_costs = [scores[i] for i in ga_order]
    x = np.arange(len(net_names))

    ax.bar(x - 0.15, gnn_costs, width=0.3, label="GNN order", alpha=0.7)
    ax.bar(x + 0.15, ga_costs, width=0.3, label="GA order", alpha=0.7)

    ax.set_title(f"Net Ordering Comparison — {board}")
    ax.set_xlabel("Net Index")
    ax.set_ylabel("Predicted Routing Cost")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{board}_ordering.png"))
    plt.close()


# ===============================================================
#  Core Prediction + Optimization
# ===============================================================
def predict_and_optimize(model_path="../results/net_gnn_model.pt",
                         data_dir="../data",
                         out_csv="../results/net_orderings.csv",
                         out_json_dir="../results/orderings_json",
                         use_ga=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Loading trained GNN from {model_path}")

    model_state = torch.load(model_path, map_location=device)
    try:
        in_dim = model_state["conv1.lin_l.weight"].shape[1]
    except Exception:
        in_dim = 10  # Fallback for your 10-feature graph_builder

    model = NetCostGNN(in_dim=in_dim).to(device)
    model.load_state_dict(model_state)
    model.eval()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(out_json_dir, exist_ok=True)
    results = []

    pcb_files = [f for f in os.listdir(data_dir) if f.endswith(".kicad_pcb")]
    if not pcb_files:
        print(" No .kicad_pcb files found.")
        return

    for pcb in pcb_files:
        print(f"\n Processing {pcb} ...")
        try:
            meta = parse_kicad(os.path.join(data_dir, pcb))
            g = build_net_graph(meta)
            g = g.to(device)

            with torch.no_grad():
                preds, h = model(g.x, g.edge_index, g.edge_attr)
                scores = preds.cpu().numpy()
                net_names = g.net_names

            order_gnn = np.argsort(scores)
            gnn_cost = np.sum(scores[order_gnn])

            if use_ga and len(scores) > 3:
                best_order, best_score = ga_optimize(scores)
                print(f" GA improved cost {best_score:.2f} (vs {gnn_cost:.2f})")
                final_order = best_order
            else:
                final_order, best_score = order_gnn, gnn_cost

            # --- Save JSON per PCB ---
            json_data = []
            for rank, idx in enumerate(final_order):
                entry = {
                    "rank": int(rank),
                    "net": net_names[idx],
                    "predicted_cost": float(scores[idx]),
                }
                json_data.append(entry)

                # Add to overall CSV list
                results.append({
                    "board": pcb,
                    "rank": rank,
                    "net": net_names[idx],
                    "pred_cost": float(scores[idx]),
                })

            json_path = os.path.join(out_json_dir, pcb.replace(".kicad_pcb", "_ordering.json"))
            with open(json_path, "w") as jf:
                json.dump({
                    "board": pcb,
                    "num_nets": len(net_names),
                    "optimized_order": json_data,
                    "ga_used": use_ga
                }, jf, indent=2)

            # Plot cost comparison
            if use_ga and len(scores) > 3:
                plot_net_ordering(pcb.replace(".kicad_pcb", ""), net_names, scores, order_gnn, final_order)

            print(f" Saved ordering for {pcb}: {len(final_order)} nets → {json_path}")

        except Exception as e:
            print(f" Error processing {pcb}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\n Saved combined CSV to {out_csv}")
    print(f" Individual JSONs in: {out_json_dir}")
    print(" Plots saved to ../results/plots/")


# ===============================================================
#  Main
# ===============================================================
if __name__ == "__main__":
    predict_and_optimize(use_ga=True)
