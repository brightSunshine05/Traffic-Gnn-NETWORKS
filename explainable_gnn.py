import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax


# ============================================================
# Edge Attention Convolution Layer
# ============================================================

class EdgeAttentionConv(nn.Module):
    """
    Graph convolution where each edge gets an attention score in [0,1].

    For each edge (i -> j):
        score = sigmoid(MLP([x_i || x_j || edge_attr]))
        message = score * W * x_j
        x_i' = x_i + sum_{j in N(i)} message

    The attention scores are stored per forward pass and can be
    retrieved for explanation.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_attr_dim: int = 1):
        super().__init__()

        self.lin_node = nn.Linear(in_dim, out_dim, bias=False)

        # Attention MLP: [x_i || x_j || edge_attr] -> scalar
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_attr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.bias = nn.Parameter(torch.zeros(out_dim))
        self._last_attn_scores = None  # stored for retrieval

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        x:          [N, in_dim]
        edge_index: [2, E]
        edge_attr:  [E, edge_attr_dim] or None

        Returns:    [N, out_dim] updated node features
        """
        src, dst = edge_index[0], edge_index[1]

        # Linear transform of source nodes
        x_transformed = self.lin_node(x)  # [N, out_dim]

        # Build attention input: [x_src || x_dst || edge_attr]
        attn_in_parts = [x[src], x[dst]]
        if edge_attr is not None:
            attn_in_parts.append(edge_attr)
        else:
            # Use zero dummy if no edge attr
            attn_in_parts.append(torch.zeros(src.size(0), 1, device=x.device))

        attn_in = torch.cat(attn_in_parts, dim=1)  # [E, 2*in + ea_dim]

        # Compute attention scores
        attn_scores = self.attn_mlp(attn_in).squeeze(-1)  # [E]
        self._last_attn_scores = attn_scores.detach()      # store for retrieval

        # Weighted message passing
        # message from j -> i: attn_score * x_j_transformed
        messages = attn_scores.unsqueeze(-1) * x_transformed[dst]  # [E, out_dim]

        # Aggregate (sum) messages at each destination node
        out = torch.zeros_like(x_transformed)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Residual connection + bias
        out = out + x_transformed + self.bias
        return out

    def get_attention_scores(self):
        """Return attention scores from the last forward pass."""
        return self._last_attn_scores


# ============================================================
# Intrinsically Explainable Traffic GNN
# ============================================================

class ExplainableTrafficGNN(nn.Module):
    """
    Full GNN using EdgeAttentionConv at every layer.

    Key capabilities:
    1. classify(data)        -> class logits
    2. explain(data)         -> dict with top-K edges and their scores
    3. explanation_loss()    -> regularization to encourage sparsity
                                (few edges dominate each classification)
    """

    def __init__(self, input_dim=1500, hidden_dim=128,
                 num_classes=4, edge_attr_dim=1,
                 sparsity_lambda=1e-3):
        super().__init__()

        self.sparsity_lambda = sparsity_lambda
        self.hidden_dim = hidden_dim

        self.conv1 = EdgeAttentionConv(input_dim,  hidden_dim, edge_attr_dim)
        self.conv2 = EdgeAttentionConv(hidden_dim, hidden_dim, edge_attr_dim)
        self.conv3 = EdgeAttentionConv(hidden_dim, hidden_dim, edge_attr_dim)
        self.conv4 = EdgeAttentionConv(hidden_dim, hidden_dim, edge_attr_dim)
        self.conv5 = EdgeAttentionConv(hidden_dim, hidden_dim, edge_attr_dim)

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

        # Store all layer attention scores for retrieval
        self._conv_layers = [self.conv1, self.conv2, self.conv3,
                             self.conv4, self.conv5]

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)

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

    def explanation_loss(self) -> torch.Tensor:
        """
        Sparsity regularizer: encourages most attention scores to be
        near 0 so that only a small subset of edges drives predictions.
        L1 penalty on all attention scores across all layers.
        This is jointly minimized with classification loss.
        """
        total = torch.tensor(0.0, requires_grad=True)
        for conv in self._conv_layers:
            scores = conv._last_attn_scores
            if scores is not None:
                total = total + scores.abs().mean()
        return self.sparsity_lambda * total

    def explain(self, data, top_k: int = 10) -> dict:
        """
        Run forward pass and return the top-K most attended edges.

        Returns:
            {
              'predicted_class': int,
              'class_name': str,
              'top_edges': list of (src_node, dst_node, attention_score),
              'layer_scores': list of tensors (one per conv layer)
            }
        """
        CLASS_NAMES = {0: "chat", 1: "email", 2: "streaming", 3: "voip"}

        self.eval()
        with torch.no_grad():
            logits = self.forward(data)

        pred_class = int(logits.argmax(dim=1).item())

        # Aggregate attention scores across all 5 layers (mean)
        layer_scores = []
        for conv in self._conv_layers:
            s = conv._last_attn_scores
            if s is not None:
                layer_scores.append(s.cpu())

        if not layer_scores:
            return {"predicted_class": pred_class, "top_edges": []}

        # Mean attention across layers: [E]
        mean_scores = torch.stack(layer_scores, dim=0).mean(dim=0)

        edge_index = data.edge_index.cpu()
        k = min(top_k, mean_scores.size(0))
        top_indices = mean_scores.topk(k).indices

        top_edges = [
            (int(edge_index[0, idx]), int(edge_index[1, idx]),
             float(mean_scores[idx]))
            for idx in top_indices
        ]

        return {
            "predicted_class": pred_class,
            "class_name": CLASS_NAMES.get(pred_class, str(pred_class)),
            "confidence": float(F.softmax(logits, dim=1).max().item()),
            "top_edges": top_edges,
            "layer_scores": layer_scores,
        }


# ============================================================
# Training function with sparsity regularization
# ============================================================

def train_with_explanation_loss(model: ExplainableTrafficGNN,
                                 loader,
                                 optimizer,
                                 device: torch.device):
    """
    Training step that adds the explanation sparsity loss to the
    classification loss. This is the key difference from standard training.
    """
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        logits = model(data)
        cls_loss  = criterion(logits, data.y)
        expl_loss = model.explanation_loss()   # sparsity regularizer
        loss = cls_loss + expl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------------
# Quick test
# -----------------------------------
if __name__ == "__main__":
    from graph_builder import flow_to_graph
    from torch_geometric.loader import DataLoader

    np.random.seed(0)
    graphs = []
    for label in range(4):
        for _ in range(4):
            flow = np.random.rand(8, 1500).astype(np.float32)
            g = flow_to_graph(flow, label)
            if g is not None:
                graphs.append(g)

    model  = ExplainableTrafficGNN(input_dim=1500, hidden_dim=32, num_classes=4)
    device = torch.device("cpu")
    loader = DataLoader(graphs, batch_size=4)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_with_explanation_loss(model, loader, opt, device)
    print(f"Training loss (cls + sparsity): {loss:.4f}")

    # Explain a single graph
    single = graphs[0]
    explanation = model.explain(single)
    print(f"\nExplanation:")
    print(f"  Class   : {explanation['class_name']} "
          f"(confidence={explanation['confidence']:.3f})")
    print(f"  Top edges (src -> dst, attention score):")
    for src, dst, score in explanation["top_edges"][:5]:
        print(f"    {src:3d} -> {dst:3d}  score={score:.4f}")
