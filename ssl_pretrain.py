import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import TrafficGNN


# ============================================================
# Augmentation functions
# ============================================================

def augment_edge_drop(data: Data, drop_ratio: float = 0.3) -> Data:
    """
    Randomly drop SIMILARITY edges (edge_attr[:,0] < 1.0).
    Sequential edges (edge_attr[:,0] == 1.0) are never dropped.
    Preserves temporal ordering while perturbing content structure.
    """
    edge_index = data.edge_index
    edge_attr  = data.edge_attr  # [E, 1] or [E, 4]

    if edge_attr is None:
        # Fall back to dropping any edge
        n_edges = edge_index.size(1)
        keep = torch.rand(n_edges) > drop_ratio
        return Data(
            x=data.x,
            edge_index=edge_index[:, keep],
            edge_attr=None,
            y=data.y
        )

    # Identify sequential edges (weight == 1.0 in first attr column)
    is_sequential = (edge_attr[:, 0] == 1.0)

    # Only drop from similarity edges
    keep_mask = is_sequential.clone()
    sim_indices = (~is_sequential).nonzero(as_tuple=True)[0]

    if len(sim_indices) > 0:
        keep_sim = torch.rand(len(sim_indices)) > drop_ratio
        keep_mask[sim_indices[keep_sim]] = True

    return Data(
        x=data.x,
        edge_index=edge_index[:, keep_mask],
        edge_attr=edge_attr[keep_mask],
        y=data.y,
        batch=getattr(data, 'batch', None)
    )


def augment_node_mask(data: Data, mask_ratio: float = 0.15) -> Data:
    """
    Randomly zero out mask_ratio fraction of each node's feature values.
    Simulates partial packet loss or byte-level obfuscation.
    """
    x = data.x.clone()
    mask = torch.rand_like(x) < mask_ratio
    x[mask] = 0.0
    return Data(
        x=x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        y=data.y,
        batch=getattr(data, 'batch', None)
    )


def augment_subgraph(data: Data, keep_ratio: float = 0.7) -> Data:
    """
    Sample a contiguous temporal subgraph by keeping the first
    keep_ratio fraction of packets (nodes). Removes all edges
    involving dropped nodes.
    """
    n = data.x.size(0)
    keep_n = max(2, int(n * keep_ratio))
    keep_nodes = torch.arange(keep_n)

    # Remap edges to only kept nodes
    mask = (data.edge_index[0] < keep_n) & (data.edge_index[1] < keep_n)
    new_edge_index = data.edge_index[:, mask]
    new_edge_attr  = data.edge_attr[mask] if data.edge_attr is not None else None

    return Data(
        x=data.x[:keep_n],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        y=data.y,
        batch=getattr(data, 'batch', None)
    )


def augment_feature_jitter(data: Data, std: float = 0.01) -> Data:
    """
    Add small Gaussian noise to normalized packet byte values.
    Noise is clipped to keep values in [0, 1].
    """
    x = data.x.clone()
    noise = torch.randn_like(x) * std
    x = torch.clamp(x + noise, 0.0, 1.0)
    return Data(
        x=x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        y=data.y,
        batch=getattr(data, 'batch', None)
    )


AUGMENTATIONS = [
    augment_edge_drop,
    augment_node_mask,
    augment_subgraph,
    augment_feature_jitter,
]


def random_augment(data: Data) -> Data:
    """Apply one randomly chosen augmentation."""
    aug_fn = AUGMENTATIONS[torch.randint(len(AUGMENTATIONS), (1,)).item()]
    return aug_fn(data)


# ============================================================
# Projection head (standard in contrastive learning)
# Maps graph embedding -> contrastive space
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# NT-Xent contrastive loss (SimCLR)
# ============================================================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.5) -> torch.Tensor:
    """
    Normalized temperature-scaled cross entropy loss.
    z1, z2: [B, proj_dim] embeddings of two augmented views.
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate: [2B, proj_dim]
    z = torch.cat([z1, z2], dim=0)

    # Similarity matrix: [2B, 2B]
    sim = torch.mm(z, z.t()) / temperature

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])

    # Mask self-similarity (diagonal)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    loss = F.cross_entropy(sim, labels)
    return loss


# ============================================================
# Pre-training loop
# ============================================================

class GraphContrastivePretrainer:
    """
    Wraps a TrafficGNN encoder + projection head for contrastive pre-training.

    Usage:
        pretrainer = GraphContrastivePretrainer(encoder)
        pretrainer.pretrain(unlabeled_dataset, epochs=100)
        # Then fine-tune encoder on labeled data using train.py
    """

    def __init__(self,
                 encoder: TrafficGNN,
                 proj_dim: int = 128,
                 temperature: float = 0.5,
                 device: torch.device = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = encoder.to(self.device)

        # Embedding dim = 5 * hidden_dim * 2 (from TrafficGNN)
        embed_dim = 5 * encoder.hidden_dim * 2
        self.projector = ProjectionHead(embed_dim, proj_dim).to(self.device)
        self.temperature = temperature

    def pretrain(self,
                 unlabeled_dataset: list,
                 epochs: int = 100,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 save_path: str = "data/pretrained_encoder.pt"):
        """
        Pre-train on unlabeled_dataset (list of PyG Data objects).
        Labels are not used during this phase.
        """
        params = list(self.encoder.parameters()) + \
                 list(self.projector.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)

        loader = DataLoader(unlabeled_dataset,
                            batch_size=batch_size,
                            shuffle=True)

        print(f"Starting contrastive pre-training | "
              f"graphs={len(unlabeled_dataset)} epochs={epochs} "
              f"device={self.device}")

        for epoch in range(1, epochs + 1):
            self.encoder.train()
            self.projector.train()
            epoch_loss = 0.0

            for batch in loader:
                # Create two augmented views
                batch1 = _apply_aug_to_batch(batch, random_augment)
                batch2 = _apply_aug_to_batch(batch, random_augment)

                batch1 = batch1.to(self.device)
                batch2 = batch2.to(self.device)

                # Get embeddings from encoder
                h1 = self.encoder.get_embedding(batch1)
                h2 = self.encoder.get_embedding(batch2)

                # Project to contrastive space
                z1 = self.projector(h1)
                z2 = self.projector(h2)

                loss = nt_xent_loss(z1, z2, self.temperature)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
                optimizer.step()

                epoch_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d} | "
                      f"Contrastive loss: {epoch_loss/len(loader):.4f}")

        # Save the pre-trained encoder weights only (not projector)
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(self.encoder.state_dict(), save_path)
        print(f"Pre-trained encoder saved to: {save_path}")

    def fine_tune(self,
                  labeled_dataset: list,
                  pretrained_path: str = "data/pretrained_encoder.pt",
                  epochs: int = 100,
                  batch_size: int = 64,
                  lr: float = 5e-4):
        """
        Load pre-trained encoder and fine-tune on labeled data.
        Returns the fine-tuned model.
        """
        self.encoder.load_state_dict(
            torch.load(pretrained_path, weights_only=True)
        )
        print(f"Loaded pre-trained encoder from: {pretrained_path}")

        optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=lr, weight_decay=1e-5
        )
        criterion = torch.nn.CrossEntropyLoss()
        loader = DataLoader(labeled_dataset,
                            batch_size=batch_size,
                            shuffle=True)

        for epoch in range(1, epochs + 1):
            self.encoder.train()
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                out  = self.encoder(batch)
                loss = criterion(out, batch.y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), max_norm=2.0
                )
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Fine-tune epoch {epoch:03d} | "
                      f"Loss: {total_loss/len(loader):.4f}")

        return self.encoder


def _apply_aug_to_batch(batch, aug_fn):
    """
    Apply aug_fn to each graph in a batched Data object.
    Returns a new batched Data (re-assembled via DataLoader).
    """
    from torch_geometric.data import Batch

    # Unbatch, augment each graph, re-batch
    graphs = batch.to_data_list()
    aug_graphs = [aug_fn(g) for g in graphs]
    return Batch.from_data_list(aug_graphs)


# -----------------------------------
# Quick test
# -----------------------------------
if __name__ == "__main__":
    from graph_builder import flow_to_graph

    np.random.seed(0)

    # Create dummy unlabeled dataset
    unlabeled = []
    for _ in range(32):
        flow = np.random.rand(8, 1500).astype(np.float32)
        g = flow_to_graph(flow, label=0)
        if g is not None:
            unlabeled.append(g)

    encoder = TrafficGNN(input_dim=1500, hidden_dim=128, num_classes=4)
    pretrainer = GraphContrastivePretrainer(encoder)

    pretrainer.pretrain(unlabeled, epochs=5, batch_size=8,
                        save_path="data/pretrained_encoder.pt")

    print("\nRunning augmentation tests...")
    g = unlabeled[0]
    for aug in AUGMENTATIONS:
        ag = aug(g)
        print(f"  {aug.__name__}: nodes={ag.num_nodes} edges={ag.num_edges}")
