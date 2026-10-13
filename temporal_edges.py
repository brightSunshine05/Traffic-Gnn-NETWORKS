import torch
import numpy as np
from torch_geometric.data import Data
from graph_builder import create_sequential_edges, create_similarity_edges

# Sentinel for missing timestamps
_NO_TIME = -1.0


# -----------------------------------
# Compute temporal edge attributes
# -----------------------------------
def compute_temporal_edge_attrs(timestamps: np.ndarray,
                                 packet_sizes: np.ndarray) -> dict:
    """
    Given timestamps and sizes for packets in a flow, compute per-pair
    temporal statistics for use as edge attributes.

    Returns a dict keyed by (i, j) -> np.array([iat, byte_rate, iat_norm])
    where:
        iat       = |t_j - t_i|  (inter-arrival time in seconds)
        byte_rate = size_j / max(iat, 1e-6)  (bytes per second)
        iat_norm  = (iat - mean_iat) / (std_iat + 1e-8)  (z-score)
    """
    n = len(timestamps)
    if n < 2:
        return {}

    # Compute pairwise IAT for sequential pairs (i, i+1)
    iats = np.abs(np.diff(timestamps))  # [n-1] sequential IATs
    mean_iat = iats.mean() if len(iats) > 0 else 0.0
    std_iat  = iats.std()  if len(iats) > 1 else 1.0

    def temporal_attr(i, j):
        iat = abs(float(timestamps[j]) - float(timestamps[i]))
        br  = float(packet_sizes[j]) / max(iat, 1e-6)
        iat_norm = (iat - mean_iat) / (std_iat + 1e-8)
        # Clamp to [-5, 5] to avoid extreme outliers
        iat_norm = float(np.clip(iat_norm, -5.0, 5.0))
        # Normalize iat to [0,1] using log scale (traffic can span ms to seconds)
        iat_log = float(np.log1p(iat * 1000) / np.log1p(1000))  # ms scale
        br_norm = float(np.log1p(br) / np.log1p(1e7))           # up to 10MB/s
        return np.array([iat_log, br_norm, iat_norm], dtype=np.float32)

    attrs = {}
    for i in range(n):
        for j in range(i + 1, n):
            attrs[(i, j)] = temporal_attr(i, j)

    return attrs


# -----------------------------------
# Build temporally-enriched graph
# -----------------------------------
def flow_to_temporal_graph(flow_vectors: np.ndarray,
                            timestamps: np.ndarray,
                            packet_sizes: np.ndarray,
                            label: int) -> Data:
    """
    Builds a PyG graph with 3 types of edges, each carrying a
    3-dimensional attribute vector [similarity, iat_log, byte_rate_norm, iat_norm].

    Sequential edges: attr = [1.0, iat, br, iat_norm]
    Similarity edges: attr = [cos_score, iat, br, iat_norm]
    """
    node_features = np.array(flow_vectors, dtype=np.float32)
    n = len(node_features)

    if n < 2:
        return None

    # Temporal attributes for each pair
    temp_attrs = compute_temporal_edge_attrs(timestamps, packet_sizes)

    all_edges   = []
    all_weights = []   # for backward compat with GNN edge_attr [E, 4]

    # ---- Sequential edges ----
    seq_e, seq_w = create_sequential_edges(n)
    for (src, dst), w in zip(seq_e, seq_w):
        t_attr = temp_attrs.get((src, dst), np.zeros(3, dtype=np.float32))
        all_edges.append([src, dst])
        all_weights.append(np.concatenate([[w], t_attr]))  # [4]

    # ---- Similarity edges ----
    sim_e, sim_w = create_similarity_edges(node_features)
    for (src, dst), w in zip(sim_e, sim_w):
        t_attr = temp_attrs.get((src, dst), np.zeros(3, dtype=np.float32))
        all_edges.append([src, dst])
        all_weights.append(np.concatenate([[w], t_attr]))  # [4]

    if len(all_edges) == 0:
        return None

    edge_index = torch.tensor(all_edges,   dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(
        np.array(all_weights, dtype=np.float32), dtype=torch.float
    )  # [E, 4]
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([label],        dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# -----------------------------------
# Temporal-aware GNN model
# Uses edge attributes in GraphConv
# -----------------------------------
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import NNConv


class TemporalTrafficGNN(nn.Module):
    """
    Uses NNConv (edge-conditioned convolution) so the 4-dim edge
    attributes are used to compute per-edge transformation matrices.

    This allows the model to:
      - Weight message passing differently for sequential vs similarity edges
      - Explicitly use IAT and byte rate when aggregating neighbor messages
    """

    def __init__(self, input_dim=1500, hidden_dim=64,
                 edge_attr_dim=4, num_classes=4):
        super().__init__()

        # Edge network: maps edge_attr -> transformation matrix [in * out]
        def make_edge_net(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(edge_attr_dim, 32),
                nn.ReLU(),
                nn.Linear(32, in_dim * out_dim)
            )

        self.conv1 = NNConv(input_dim, hidden_dim,
                            make_edge_net(input_dim, hidden_dim), aggr='mean')
        self.conv2 = NNConv(hidden_dim, hidden_dim,
                            make_edge_net(hidden_dim, hidden_dim), aggr='mean')
        self.conv3 = NNConv(hidden_dim, hidden_dim,
                            make_edge_net(hidden_dim, hidden_dim), aggr='mean')
        self.conv4 = NNConv(hidden_dim, hidden_dim,
                            make_edge_net(hidden_dim, hidden_dim), aggr='mean')
        self.conv5 = NNConv(hidden_dim, hidden_dim,
                            make_edge_net(hidden_dim, hidden_dim), aggr='mean')

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

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        embeddings = []

        def pool(h):
            return torch.cat([global_max_pool(h, batch),
                              global_mean_pool(h, batch)], dim=1)

        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        embeddings.append(pool(x))

        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        embeddings.append(pool(x))

        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        embeddings.append(pool(x))

        x = F.relu(self.bn4(self.conv4(x, edge_index, edge_attr)))
        embeddings.append(pool(x))

        x = F.relu(self.bn5(self.conv5(x, edge_index, edge_attr)))
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
    np.random.seed(0)
    n_pkts = 8
    flow = np.random.rand(n_pkts, 1500).astype(np.float32)

    # Simulate timestamps with realistic inter-arrival times (~10ms spacing)
    timestamps  = np.cumsum(np.random.exponential(0.01, n_pkts))
    packet_sizes = np.random.randint(64, 1500, n_pkts).astype(np.float32)

    g = flow_to_temporal_graph(flow, timestamps, packet_sizes, label=2)

    print("Temporal graph:")
    print(f"  Nodes      : {g.num_nodes}")
    print(f"  Edges      : {g.num_edges}")
    print(f"  edge_attr  : {g.edge_attr.shape}  (should be [E, 4])")
    print(f"  Sample attr: {g.edge_attr[0]}")

    # Check the model runs
    from torch_geometric.loader import DataLoader
    loader = DataLoader([g, g], batch_size=2)
    model  = TemporalTrafficGNN()
    for batch_data in loader:
        out = model(batch_data)
        print(f"  Model output: {out.shape}")
        break
