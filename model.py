import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d, Linear, Dropout


class TrafficGNN(torch.nn.Module):
    def __init__(self, input_dim=1500, hidden_dim=128, num_classes=4):
        super(TrafficGNN, self).__init__()

        self.hidden_dim = hidden_dim

        # 5 GraphConv layers (paper eq.7)
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)

        # BatchNorm after each layer (paper: stabilize training, mitigate overfitting)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.bn4 = BatchNorm1d(hidden_dim)
        self.bn5 = BatchNorm1d(hidden_dim)

        # Each layer contributes (hidden_dim * 2) to the final embedding
        # 5 layers x (max_pool + mean_pool) = 5 * hidden_dim * 2
        concat_dim = 5 * hidden_dim * 2  # = 5 * 128 * 2 = 1280

        # 3-layer MLP (paper section 3.3)
        self.lin1 = Linear(concat_dim, 256)
        self.lin2 = Linear(256, 128)
        self.lin3 = Linear(128, num_classes)

        self.dropout = Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Collect per-layer pooled embeddings
        layer_embeddings = []

        # Layer 1
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        layer_embeddings.append(torch.cat([x_max, x_mean], dim=1))

        # Layer 2
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        layer_embeddings.append(torch.cat([x_max, x_mean], dim=1))

        # Layer 3
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        layer_embeddings.append(torch.cat([x_max, x_mean], dim=1))

        # Layer 4
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        layer_embeddings.append(torch.cat([x_max, x_mean], dim=1))

        # Layer 5
        x = F.relu(self.bn5(self.conv5(x, edge_index)))
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        layer_embeddings.append(torch.cat([x_max, x_mean], dim=1))

        # Concatenate all 5 layer embeddings (paper section 3.3)
        graph_embed = torch.cat(layer_embeddings, dim=1)  # [batch, 1280]

        # MLP classifier
        out = F.relu(self.lin1(graph_embed))
        out = self.dropout(out)
        out = F.relu(self.lin2(out))
        out = self.dropout(out)
        out = self.lin3(out)

        return out

    def get_embedding(self, data):
        """Return graph-level embedding before the classifier (for SSL pre-training)."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        layer_embeddings = []

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        layer_embeddings.append(torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1))

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        layer_embeddings.append(torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1))

        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        layer_embeddings.append(torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1))

        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        layer_embeddings.append(torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1))

        x = F.relu(self.bn5(self.conv5(x, edge_index)))
        layer_embeddings.append(torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1))

        return torch.cat(layer_embeddings, dim=1)