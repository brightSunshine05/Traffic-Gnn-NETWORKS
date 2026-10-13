import torch
import numpy as np
from torch_geometric.data import Data

from graph_builder import create_sequential_edges
from adaptive_threshold import extract_flow_stats, soft_edge_mask


def flow_to_candidate_graph(flow_vectors: np.ndarray, label: int) -> Data:
    node_features = np.array(flow_vectors, dtype=np.float32)
    n = node_features.shape[0]
    if n < 2:
        return None

    stats, sim_matrix, _ = extract_flow_stats(node_features)

    seq_e, seq_w = create_sequential_edges(n)
    all_edges = list(seq_e)
    all_sim = [1.0] * len(seq_e)          # sequential edges: sim placeholder, never gated
    all_is_seq = [1.0] * len(seq_e)

    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append([i, j])
            all_sim.append(float(sim_matrix[i, j]))
            all_is_seq.append(0.0)

    if len(all_edges) == 0:
        return None

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    sim_attr = torch.tensor(all_sim, dtype=torch.float)
    is_seq = torch.tensor(all_is_seq, dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)
    flow_stats = torch.tensor(stats, dtype=torch.float).unsqueeze(0)  # [1, 10]

    return Data(x=x, edge_index=edge_index, sim_attr=sim_attr, is_seq=is_seq,
                y=y, flow_stats=flow_stats)


# -----------------------------------
# Adaptive-threshold GNN: gates candidate
# edges with a per-flow learned threshold
# before every GraphConv layer
# -----------------------------------
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool
from adaptive_threshold import AdaptiveThresholdNet


class AdaptiveTrafficGNN(nn.Module):
    """
    Same 5-layer GraphConv + BatchNorm + multi-layer-pool architecture as
    model.TrafficGNN, but every edge is weighted by a per-flow, per-edge
    gate: sequential edges always pass with weight 1; similarity edges pass
    with soft_edge_mask(sim, theta_hat), where theta_hat is predicted by
    AdaptiveThresholdNet from the flow's own statistics. This is the
    mechanism Novelty 1 is supposed to implement end-to-end.
    """

    def __init__(self, input_dim=1500, hidden_dim=128, num_classes=4,
                 stats_dim=10, temperature=10.0):
        super().__init__()
        self.threshold_net = AdaptiveThresholdNet(stats_dim=stats_dim, hidden=32)
        self.temperature = temperature

        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)

        concat_dim = 5 * hidden_dim * 2
        self.lin1 = nn.Linear(concat_dim, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.last_theta = None  # for logging / audit, mean theta of last batch

    def _edge_weight(self, data):
        # theta per candidate edge, broadcast from per-graph flow_stats via data.batch
        theta_per_graph = self.threshold_net(data.flow_stats).squeeze(-1)   # [num_graphs]
        edge_batch = data.batch[data.edge_index[0]]                        # [E] -> graph id
        theta_per_edge = theta_per_graph[edge_batch]                        # [E]

        soft_gate = soft_edge_mask(data.sim_attr, theta_per_edge, self.temperature)
        # sequential edges (is_seq==1) always pass at full weight;
        # similarity edges are gated by the learned per-flow threshold
        edge_weight = data.is_seq + (1.0 - data.is_seq) * soft_gate
        self.last_theta = theta_per_graph.detach()
        return edge_weight

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = self._edge_weight(data)

        embeddings = []

        def pool(h):
            return torch.cat([global_max_pool(h, batch), global_mean_pool(h, batch)], dim=1)

        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        embeddings.append(pool(x))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight)))
        embeddings.append(pool(x))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_weight)))
        embeddings.append(pool(x))
        x = F.relu(self.bn4(self.conv4(x, edge_index, edge_weight)))
        embeddings.append(pool(x))
        x = F.relu(self.bn5(self.conv5(x, edge_index, edge_weight)))
        embeddings.append(pool(x))

        g = torch.cat(embeddings, dim=1)
        out = F.relu(self.lin1(g))
        out = self.dropout(out)
        out = F.relu(self.lin2(out))
        out = self.dropout(out)
        return self.lin3(out)


# -----------------------------------
# Quick test
# -----------------------------------
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader

    np.random.seed(0)
    graphs = []
    for label in range(4):
        for _ in range(4):
            flow = np.random.rand(8, 1500).astype(np.float32)
            g = flow_to_candidate_graph(flow, label)
            if g is not None:
                graphs.append(g)

    model = AdaptiveTrafficGNN(input_dim=1500, hidden_dim=32, num_classes=4)
    loader = DataLoader(graphs, batch_size=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()

    for batch in loader:
        opt.zero_grad()
        out = model(batch)
        loss = crit(out, batch.y)
        loss.backward()
        opt.step()
        print(f"loss={loss.item():.4f}  mean theta this batch={model.last_theta.mean().item():.4f}")
        # confirm gradient actually reaches the threshold net
        grad_norm = sum(p.grad.norm().item() for p in model.threshold_net.parameters() if p.grad is not None)
        print(f"threshold_net grad norm: {grad_norm:.6f}  (should be > 0)")
        break
