import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=64, num_heads=8, dropout=0.5):
        super(GAT, self).__init__()

        self.gat1 = GATConv(num_node_features, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=False, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.gat3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.log_softmax(self.fc(x), dim=-1)
        return x