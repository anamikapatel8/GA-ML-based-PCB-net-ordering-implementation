# train_gnn.py 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np


# -------------------------------------------------
#  GNN Model Definition
# -------------------------------------------------
class NetCostGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        out = self.lin(h)
        return out.squeeze(), h   # out is per-node cost


# -------------------------------------------------
#  Load Graphs + Routing Cost Labels
# -------------------------------------------------
def load_graphs_with_labels(
    graph_dir="../graphs",
    csv_path="../results/per_net_dataset1.csv"
):
    from torch.serialization import add_safe_globals
    from torch_geometric.data import Data
    add_safe_globals([Data])

    df = pd.read_csv(csv_path)
    df["board_name"] = df["board_name"].str.replace(".kicad_pcb", "", regex=False)

    graphs = []

    for f in os.listdir(graph_dir):
        if not f.endswith(".pt"):
            continue

        pcb_name = f.replace(".pt", "")
        matching_rows = df[df["board_name"] == pcb_name]

        if matching_rows.empty:
            print(f"[WARN] No labels found for {pcb_name}, skipping.")
            continue

        # Load PyG graph
        g = torch.load(os.path.join(graph_dir, f), weights_only=False)

        # Ensure graph stores its board name (used for embeddings)
        g.board_name = pcb_name

        # Build per-node labels
        y = []
        for net in g.net_names:
            rows = matching_rows[matching_rows["net"] == net]
            if rows.empty:
                y.append(0.0)
            else:
                y.append(rows["est_routed_len"].mean())

        g.y = torch.tensor(y, dtype=torch.float)
        graphs.append(g)

        print(f"Loaded {pcb_name}: nodes={len(g.x)}, labels={len(g.y)}, x_dim={g.x.size(1)}")

    return graphs


# -------------------------------------------------
#  Training Loop
# -------------------------------------------------
def train(model, loader, device, epochs=80):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()

            pred, _ = model(data.x, data.edge_index, data.edge_attr)
            loss = loss_fn(pred, data.y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:03d} | Loss = {total_loss / len(loader):.4f}")


# -------------------------------------------------
#  Main Script
# -------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = load_graphs_with_labels()

    if not graphs:
        raise RuntimeError("No valid graphs found in ../graphs")

    loader = DataLoader(graphs, batch_size=1, shuffle=True)

    in_dim = graphs[0].x.size(1)
    print(f"\n Training NetCostGNN on {len(graphs)} graphs | input_dim={in_dim}")

    model = NetCostGNN(in_dim=in_dim).to(device)
    train(model, loader, device, epochs=80)

    # -------------------------------------------------
    # Save PyTorch model weights
    # -------------------------------------------------
    os.makedirs("../results", exist_ok=True)
    torch.save(model.state_dict(), "../results/net_gnn_model.pt")
    print("Saved PyTorch model to ../results/net_gnn_model.pt")

    # -------------------------------------------------
    # Save TorchScript JIT model (C++ Autorouter reads this)
    # -------------------------------------------------
    scripted = torch.jit.script(model)
    scripted.save("../results/net_gnn_model_jit.pt")
    print("Saved TorchScript JIT model to ../results/net_gnn_model_jit.pt")

    # -------------------------------------------------
    # Save embeddings for each board
    # -------------------------------------------------
    model.eval()
    os.makedirs("../embeddings", exist_ok=True)

    with torch.no_grad():
        for g in graphs:
            _, h = model(
                g.x.to(device),
                g.edge_index.to(device),
                g.edge_attr.to(device)
            )
            out_path = f"../embeddings/{g.board_name}_embeddings.npy"
            np.save(out_path, h.cpu().numpy())
            print(f"Saved embeddings → {out_path}")
