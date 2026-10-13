import torch
import numpy as np
from torch_geometric.data import Data, HeteroData
from graph_builder import (
    create_sequential_edges,
    create_similarity_edges,
    compute_similarity_threshold,
)

NGRAM_SIZE   = 150    # paper: non-overlapping n-grams of 150 bytes
NGRAMS_PER_PKT = 10  # paper: 10 n-grams per packet (10 * 150 = 1500 = MTU)
MTU          = 1500


# -----------------------------------
# Split packet vector into n-grams
# -----------------------------------
def packet_to_ngrams(packet_vec: np.ndarray,
                     ngram_size: int = NGRAM_SIZE,
                     n_grams: int = NGRAMS_PER_PKT) -> np.ndarray:
    """
    Reshape a 1500-dim packet vector into (10, 150) n-gram matrix.
    Paper section 3.2: non-overlapping n-grams with n=150.
    """
    vec = packet_vec[:ngram_size * n_grams]  # ensure exact length
    return vec.reshape(n_grams, ngram_size)   # [10, 150]


# -----------------------------------
# Build hierarchical graph
# -----------------------------------
def flow_to_hierarchical_graph(flow_vectors: np.ndarray, label: int) -> Data:
    """
    Input:
        flow_vectors: np.ndarray [num_packets, 1500]
        label: int class label

    Output:
        PyG Data object with:
          x          [total_nodes, feature_dim]  (mixed: ngrams + packets)
          edge_index [2, E]                       all edges combined
          edge_attr  [E, 1]                       edge weights
          y          [1]                          label
          node_type  [total_nodes]               0=ngram, 1=packet (for masking)
          level_ptr  [num_packets+1]             packet -> ngram index ranges
    """
    num_packets = len(flow_vectors)
    if num_packets < 2:
        return None

    # ---- Level 0: n-gram nodes ----
    ngram_nodes = []     # list of 150-dim vectors
    pkt_to_ngram_range = []  # (start_idx, end_idx) of ngrams for each packet

    for pkt_vec in flow_vectors:
        ngrams = packet_to_ngrams(pkt_vec)        # [10, 150]
        start  = len(ngram_nodes)
        for ng in ngrams:
            ngram_nodes.append(ng)
        end = len(ngram_nodes)
        pkt_to_ngram_range.append((start, end))

    num_ngram_nodes = len(ngram_nodes)
    ngram_features  = np.array(ngram_nodes, dtype=np.float32)  # [N0, 150]

    # ---- Level 1: packet nodes ----
    # Packet node feature = the full 1500-dim packet vector
    packet_features = np.array(flow_vectors, dtype=np.float32)  # [N1, 1500]

    # Pad ngram features to 1500 dims to allow a single combined x tensor
    ngram_padded = np.pad(
        ngram_features,
        ((0, 0), (0, MTU - NGRAM_SIZE)),
        mode='constant'
    )  # [N0, 1500]

    # Offset: packet nodes start at index num_ngram_nodes
    pkt_offset = num_ngram_nodes

    # Combined node feature matrix
    all_features = np.concatenate([ngram_padded, packet_features], axis=0)
    total_nodes  = all_features.shape[0]

    # Node type labels (0=ngram, 1=packet)
    node_type = np.array(
        [0] * num_ngram_nodes + [1] * num_packets,
        dtype=np.int64   # MODIFICATION (CHANGES.md item 2): np.long is not
                          # guaranteed portable across numpy versions; int64
                          # is the explicit, version-stable dtype PyTorch expects.
    )

    # ---- Build edges ----
    all_edges   = []
    all_weights = []

    # --- Within-level 0: sequential + similarity edges among n-grams ---
    # Group n-grams by their parent packet (similarity within each packet)
    for pkt_idx, (start, end) in enumerate(pkt_to_ngram_range):
        chunk = ngram_features[start:end]   # [10, 150]
        n_chunk = len(chunk)

        # Sequential edges within this packet's n-grams
        seq_e, seq_w = create_sequential_edges(n_chunk)
        for (src, dst), w in zip(seq_e, seq_w):
            all_edges.append([start + src, start + dst])
            all_weights.append(w)

        # Similarity edges within this packet's n-grams
        sim_e, sim_w = create_similarity_edges(chunk)
        for (src, dst), w in zip(sim_e, sim_w):
            all_edges.append([start + src, start + dst])
            all_weights.append(w)

    # --- Within-level 1: sequential + similarity edges among packets ---
    seq_e, seq_w = create_sequential_edges(num_packets)
    for (src, dst), w in zip(seq_e, seq_w):
        all_edges.append([pkt_offset + src, pkt_offset + dst])
        all_weights.append(w)

    sim_e, sim_w = create_similarity_edges(packet_features)
    for (src, dst), w in zip(sim_e, sim_w):
        all_edges.append([pkt_offset + src, pkt_offset + dst])
        all_weights.append(w)

    # --- Cross-level edges: each n-gram -> its parent packet ----
    # Bidirectional to allow message passing in both directions
    for pkt_idx, (start, end) in enumerate(pkt_to_ngram_range):
        pkt_node = pkt_offset + pkt_idx
        for ng_idx in range(start, end):
            # n-gram -> packet (bottom-up)
            all_edges.append([ng_idx, pkt_node])
            all_weights.append(1.0)
            # packet -> n-gram (top-down)
            all_edges.append([pkt_node, ng_idx])
            all_weights.append(1.0)

    if len(all_edges) == 0:
        return None

    edge_index = torch.tensor(all_edges,   dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(all_weights, dtype=torch.float).unsqueeze(1)
    x          = torch.tensor(all_features, dtype=torch.float)
    y          = torch.tensor([label],      dtype=torch.long)
    nt         = torch.tensor(node_type,    dtype=torch.long)

    # Store packet-to-ngram ranges for pooling later
    level_ptr = torch.tensor(
        [start for start, _ in pkt_to_ngram_range] + [num_ngram_nodes],
        dtype=torch.long
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        node_type=nt,
        level_ptr=level_ptr,
        num_ngram_nodes=num_ngram_nodes,
        num_packet_nodes=num_packets,
    )
    return data


# -----------------------------------
# Hierarchical GNN model
# Processes the unified graph and
# separately pools ngram and packet
# level nodes before classification
# -----------------------------------
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool


class HierarchicalTrafficGNN(nn.Module):
    """
    Processes the hierarchical graph in a single forward pass.

    Strategy:
    1. Run 3 GNN layers over the full graph (ngrams + packets + cross edges)
    2. Pool ngram-level nodes separately -> ngram embedding
    3. Pool packet-level nodes separately -> packet embedding
    4. Concatenate both -> classify

    This is the core novelty: the GNN sees both levels simultaneously
    and the cross-level edges let packet nodes inform n-gram processing
    and vice versa.
    """

    def __init__(self, input_dim=1500, hidden_dim=128, num_classes=4):
        super().__init__()

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

        # Two separate pooling paths -> double the embedding
        # ngram-level + packet-level, each with max+mean per 5 layers
        pooled_dim = 5 * hidden_dim * 2 * 2  # 2 levels

        self.lin1 = nn.Linear(pooled_dim, 512)
        self.lin2 = nn.Linear(512, 128)
        self.lin3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_type = data.node_type              # 0=ngram, 1=packet
        num_ngram = data.num_ngram_nodes        # scalar (per graph in batch)

        # Build separate batch vectors for each node type
        ngram_mask  = (node_type == 0)
        packet_mask = (node_type == 1)
        batch_ngram  = batch[ngram_mask]
        batch_packet = batch[packet_mask]

        ngram_embeds  = []
        packet_embeds = []

        def pool_both(h):
            h_ng = h[ngram_mask]
            h_pk = h[packet_mask]
            ng_embed = torch.cat([
                global_max_pool(h_ng, batch_ngram),
                global_mean_pool(h_ng, batch_ngram)
            ], dim=1)
            pk_embed = torch.cat([
                global_max_pool(h_pk, batch_packet),
                global_mean_pool(h_pk, batch_packet)
            ], dim=1)
            return ng_embed, pk_embed

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        ng, pk = pool_both(x)
        ngram_embeds.append(ng); packet_embeds.append(pk)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        ng, pk = pool_both(x)
        ngram_embeds.append(ng); packet_embeds.append(pk)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        ng, pk = pool_both(x)
        ngram_embeds.append(ng); packet_embeds.append(pk)

        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        ng, pk = pool_both(x)
        ngram_embeds.append(ng); packet_embeds.append(pk)

        x = F.relu(self.bn5(self.conv5(x, edge_index)))
        ng, pk = pool_both(x)
        ngram_embeds.append(ng); packet_embeds.append(pk)

        # Concatenate all layer embeddings for both levels
        graph_embed = torch.cat(ngram_embeds + packet_embeds, dim=1)

        out = F.relu(self.lin1(graph_embed))
        out = self.dropout(out)
        out = F.relu(self.lin2(out))
        out = self.dropout(out)
        out = self.lin3(out)
        return out


# -----------------------------------
# Quick test
# -----------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    flow = np.random.rand(8, 1500).astype(np.float32)

    g = flow_to_hierarchical_graph(flow, label=0)
    print("Hierarchical graph:")
    print(f"  Total nodes : {g.num_nodes}")
    print(f"  N-gram nodes: {g.num_ngram_nodes}")
    print(f"  Packet nodes: {g.num_packet_nodes}")
    print(f"  Edges       : {g.num_edges}")
    print(f"  x shape     : {g.x.shape}")
    print(f"  edge_attr   : {g.edge_attr.shape}")
    print(f"  node_type   : {g.node_type.shape}")
