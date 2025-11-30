# gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

class NetGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32, use_attention=False):
        super().__init__()
        Conv = GATConv if use_attention else SAGEConv
        self.conv1 = Conv(in_dim, hidden_dim)
        self.conv2 = Conv(hidden_dim, out_dim)
        self.fc = torch.nn.Linear(out_dim, 1)  # predict routing cost

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        cost_pred = self.fc(x)
        return cost_pred, x  # return both cost and embeddings
