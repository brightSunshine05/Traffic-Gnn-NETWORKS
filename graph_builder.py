import torch
import numpy as np
from torch_geometric.data import Data

def create_sequential_edges(num_nodes):
    edges = []
    weights = []
    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        weights.append(1.0)
    return edges, weights


# -----------------------------------
# Compute similarity threshold
# Paper Algorithm 2: compare 85th,
# 90th, 95th percentile -- if equal
# use 90th, otherwise use the max
# -----------------------------------
def compute_similarity_threshold(sim_scores):
    """
    Paper section 3.1.2 and Algorithm 2:
    theta85 = 85th percentile (rounded to 2 decimal places)
    theta90 = 90th percentile
    theta95 = 95th percentile
    If all equal -> use theta90
    Else -> use max(theta85, theta90, theta95)
    """
    if len(sim_scores) == 0:
        return 0.9  # fallback default

    theta85 = round(float(np.percentile(sim_scores, 85)), 2)
    theta90 = round(float(np.percentile(sim_scores, 90)), 2)
    theta95 = round(float(np.percentile(sim_scores, 95)), 2)

    if theta85 == theta90 == theta95:
        return theta90
    else:
        return max(theta85, theta90, theta95)


# -----------------------------------
# Similarity (trivial) edges
# Paper eq.10: edge weight = cosine
# similarity score if >= threshold
# -----------------------------------
def create_similarity_edges(node_features):
    """
    Paper section 3.1.2 (Trivial edges) and Algorithm 3:
    - Compute pairwise cosine similarity
    - Compute threshold via Algorithm 2
    - Create edge (i,j) with weight = sim[i,j] if sim >= threshold
    Returns: list of [src, dst] pairs, list of float weights
    """
    n = len(node_features)
    if n < 2:
        return [], []

    # Safely normalize vectors
    norms = np.linalg.norm(node_features, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = node_features / norms

    # Full cosine similarity matrix
    sim_matrix = np.dot(normalized, normalized.T)
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    # Extract upper triangle (excluding diagonal) for threshold computation
    rows, cols = np.triu_indices(n, k=1)
    triu_vals = sim_matrix[rows, cols]

    # Compute threshold using paper's Algorithm 2
    threshold = compute_similarity_threshold(triu_vals)

    edges = []
    weights = []

    # Paper Algorithm 3: iterate pairs and add edge if sim >= threshold
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i][j]
            if score >= threshold:
                edges.append([i, j])
                weights.append(float(score))  # store actual similarity as weight

    return edges, weights


# -----------------------------------
# Convert a single flow's packet
# vectors into a PyG Data object
# -----------------------------------
def flow_to_graph(flow_vectors, label):
    """
    Builds a PyTorch Geometric Data object from one flow.
    Nodes  = packets (feature vectors of shape [num_pkts, MTU])
    Edges  = sequential edges + similarity edges (both with weights)
    Labels = traffic class integer
    """
    node_features = np.array(flow_vectors, dtype=np.float32)
    num_nodes = node_features.shape[0]

    if num_nodes == 0:
        return None

    # Build both edge types with weights
    seq_edges, seq_weights = create_sequential_edges(num_nodes)
    sim_edges, sim_weights = create_similarity_edges(node_features)

    all_edges = seq_edges + sim_edges
    all_weights = seq_weights + sim_weights

    if len(all_edges) == 0:
        return None

    # Convert to tensors
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(all_weights, dtype=torch.float).unsqueeze(1)  # [E, 1]
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)