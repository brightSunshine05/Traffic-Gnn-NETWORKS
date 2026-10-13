import torch
import torch.nn as nn
import numpy as np

def extract_flow_stats(node_features: np.ndarray) -> np.ndarray:
   
    n = len(node_features)

    means  = node_features.mean(axis=1)   # per-packet mean
    norms  = np.linalg.norm(node_features, axis=1)

    # Pairwise cosine similarity (upper triangle only)
    safe_norms = np.where(norms == 0, 1e-8, norms)
    normalized = node_features / safe_norms[:, None]
    sim_matrix = np.clip(np.dot(normalized, normalized.T), -1.0, 1.0)
    rows, cols = np.triu_indices(n, k=1)
    sims = sim_matrix[rows, cols] if len(rows) > 0 else np.array([0.0])

    stats = np.array([
        float(means.mean()),
        float(means.std()) if n > 1 else 0.0,
        float(norms.mean()),
        float(norms.std()) if n > 1 else 0.0,
        float(sims.mean()),
        float(sims.std()) if len(sims) > 1 else 0.0,
        n / 10.0,                            # normalized packet count
        float((means < 0.01).mean()),        # fraction of near-zero packets
        float(sims.max()),
        float(sims.min()),
    ], dtype=np.float32)

    return stats, sim_matrix, sims


# -----------------------------------
# Adaptive Threshold MLP
# -----------------------------------
class AdaptiveThresholdNet(nn.Module):
    """
    Small MLP: flow_stats (10-dim) -> threshold scalar in [0, 1].

    Trained jointly with the GNN via a differentiable relaxation of the
    hard edge threshold. During training we use a soft sigmoid gate;
    at inference we threshold at 0.5 of the gate value.
    """

    def __init__(self, stats_dim: int = 10, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stats_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()            # output in (0, 1)
        )

    def forward(self, flow_stats: torch.Tensor) -> torch.Tensor:
        """
        flow_stats: [B, 10] batch of flow stat vectors
        returns:    [B, 1]  threshold per flow
        """
        return self.net(flow_stats)


# -----------------------------------
# Soft edge selection (differentiable)
# Used during training so gradients
# flow through the threshold
# -----------------------------------
def soft_edge_mask(sim_scores: torch.Tensor,
                   threshold: torch.Tensor,
                   temperature: float = 10.0) -> torch.Tensor:
    """
    Differentiable approximation of the hard threshold step.
    Uses a steep sigmoid centered at threshold.

    sim_scores: [E]    pairwise cosine similarity values
    threshold:  scalar learned threshold for this flow
    temperature: steepness of the sigmoid (higher = harder gate)
    returns:    [E]    soft mask in (0, 1)
    """
    return torch.sigmoid(temperature * (sim_scores - threshold))


# -----------------------------------
# Inference-time hard threshold
# -----------------------------------
def adaptive_threshold_edges(node_features: np.ndarray,
                              threshold_net: AdaptiveThresholdNet,
                              device: torch.device):
    """
    At inference: run threshold_net on flow stats, get a hard scalar
    threshold, then select edges whose cosine similarity >= threshold.

    Returns:
        edges   list of [src, dst] pairs
        weights list of float similarity scores
        threshold float value used (for logging/audit)
    """
    threshold_net.eval()

    stats, sim_matrix, _ = extract_flow_stats(node_features)
    stats_tensor = torch.tensor(stats, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        theta = threshold_net(stats_tensor).item()  # scalar in (0,1)

    n = len(node_features)
    edges, weights = [], []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i][j])
            if score >= theta:
                edges.append([i, j])
                weights.append(score)

    return edges, weights, theta


# -----------------------------------
# Quick test
# -----------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    dummy_flow = np.random.rand(8, 1500).astype(np.float32)

    stats, sim_matrix, sims = extract_flow_stats(dummy_flow)
    print("Flow stats shape:", stats.shape)
    print("Flow stats:", np.round(stats, 3))

    net = AdaptiveThresholdNet()
    stats_t = torch.tensor(stats).unsqueeze(0)
    theta = net(stats_t)
    print(f"Predicted threshold: {theta.item():.4f}")

    device = torch.device("cpu")
    edges, weights, theta_val = adaptive_threshold_edges(dummy_flow, net, device)
    print(f"Edges created: {len(edges)} at threshold {theta_val:.4f}")
