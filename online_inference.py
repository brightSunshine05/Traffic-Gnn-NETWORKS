import torch
import numpy as np
import time
from typing import Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from graph_builder import compute_similarity_threshold

# Default sliding window size (matches paper's truncation of 10 packets)
DEFAULT_WINDOW = 10


# ============================================================
# Incremental Flow Graph
# Maintains a live graph that grows one node at a time
# ============================================================

class IncrementalFlowGraph:
    """
    Maintains a live graph of an in-progress network flow.

    Usage:
        graph = IncrementalFlowGraph(window_size=10)
        for packet_vector in arriving_packets:
            graph.add_packet(packet_vector)
            if graph.is_ready():
                data = graph.to_pyg()
                # pass to model for classification

    Internal state:
        _nodes      : list of 1500-dim float32 arrays (sliding window)
        _seq_edges  : list of [src, dst] sequential edges
        _sim_edges  : list of [src, dst, weight] similarity edges
        _timestamps : list of arrival times (for temporal edges)
    """

    def __init__(self,
                 window_size: int = DEFAULT_WINDOW,
                 sim_percentile: int = 90,
                 min_packets_before_classify: int = 3):
        self.window_size = window_size
        self.sim_percentile = sim_percentile
        self.min_before_classify = min_packets_before_classify

        self._nodes: list = []           # [N, 1500] packet vectors
        self._timestamps: list = []      # arrival times
        self._seq_edges: list = []       # [[src, dst], ...]
        self._seq_weights: list = []
        self._sim_edges: list = []       # [[src, dst], ...]
        self._sim_weights: list = []
        self._packet_count: int = 0      # total packets seen (not just in window)

    def add_packet(self, packet_vector: np.ndarray,
                   timestamp: Optional[float] = None):
        """
        Add one new packet to the live graph.
        Maintains a sliding window of at most window_size packets.
        """
        if timestamp is None:
            timestamp = time.time()

        vec = np.array(packet_vector, dtype=np.float32).flatten()[:1500]
        if len(vec) < 1500:
            vec = np.pad(vec, (0, 1500 - len(vec)))

        self._packet_count += 1

        # --- Sliding window: if full, evict oldest packet ---
        if len(self._nodes) >= self.window_size:
            self._evict_oldest()

        new_idx = len(self._nodes)
        self._nodes.append(vec)
        self._timestamps.append(timestamp)

        # --- Sequential edge: connect to previous node ---
        if new_idx > 0:
            self._seq_edges.append([new_idx - 1, new_idx])
            self._seq_weights.append(1.0)

        # --- Similarity edges: check against all existing nodes ---
        if new_idx > 0:
            self._update_similarity_edges(new_idx)

    def _evict_oldest(self):
        """
        Remove oldest packet (index 0) from the window.
        Shift all node indices by -1 and remove stale edges.
        """
        self._nodes.pop(0)
        self._timestamps.pop(0)

        # Re-index sequential edges
        new_seq = []
        new_seq_w = []
        for (src, dst), w in zip(self._seq_edges, self._seq_weights):
            new_src = src - 1
            new_dst = dst - 1
            if new_src >= 0 and new_dst >= 0:
                new_seq.append([new_src, new_dst])
                new_seq_w.append(w)
        self._seq_edges   = new_seq
        self._seq_weights = new_seq_w

        # Re-index and prune similarity edges
        new_sim = []
        new_sim_w = []
        for (src, dst), w in zip(self._sim_edges, self._sim_weights):
            new_src = src - 1
            new_dst = dst - 1
            if new_src >= 0 and new_dst >= 0:
                new_sim.append([new_src, new_dst])
                new_sim_w.append(w)
        self._sim_edges   = new_sim
        self._sim_weights = new_sim_w

    def _update_similarity_edges(self, new_idx: int):
        """
        Compute cosine similarity of the new node to all existing nodes
        and add edges where similarity >= adaptive threshold.
        """
        if len(self._nodes) < 2:
            return

        node_features = np.array(self._nodes, dtype=np.float32)
        new_vec = node_features[new_idx]

        # Normalize
        norm_new = np.linalg.norm(new_vec)
        if norm_new < 1e-8:
            return

        # Compare new node vs all previous nodes
        scores = []
        for j in range(new_idx):
            vec_j = node_features[j]
            norm_j = np.linalg.norm(vec_j)
            if norm_j < 1e-8:
                scores.append(0.0)
            else:
                score = float(np.dot(new_vec, vec_j) / (norm_new * norm_j))
                score = np.clip(score, -1.0, 1.0)
                scores.append(score)

        if len(scores) == 0:
            return

        # Compute threshold from existing similarity scores
        threshold = compute_similarity_threshold(np.array(scores))

        for j, score in enumerate(scores):
            if score >= threshold:
                self._sim_edges.append([j, new_idx])
                self._sim_weights.append(float(score))

    def is_ready(self) -> bool:
        """True once we have enough packets to classify."""
        return len(self._nodes) >= self.min_before_classify

    def num_packets(self) -> int:
        """Current number of packets in the window."""
        return len(self._nodes)

    def to_pyg(self) -> Optional[Data]:
        """Convert current window to a PyG Data object."""
        if len(self._nodes) < 2:
            return None

        all_edges   = self._seq_edges   + self._sim_edges
        all_weights = self._seq_weights + self._sim_weights

        if len(all_edges) == 0:
            return None

        x          = torch.tensor(np.array(self._nodes), dtype=torch.float)
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(all_weights, dtype=torch.float).unsqueeze(1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def reset(self):
        """Clear all state (start of new flow)."""
        self._nodes.clear()
        self._timestamps.clear()
        self._seq_edges.clear()
        self._seq_weights.clear()
        self._sim_edges.clear()
        self._sim_weights.clear()
        self._packet_count = 0


# ============================================================
# Early-Exit Classifier
# Classifies as soon as confidence > threshold
# ============================================================

CLASS_NAMES = {0: "chat", 1: "email", 2: "streaming", 3: "voip"}


class EarlyExitClassifier:
    """
    Wraps a trained GNN model for online, early-exit classification.

    For each incoming packet:
      1. Add to IncrementalFlowGraph
      2. If min_packets reached, run GNN on current graph
      3. If max(softmax) >= confidence_threshold -> emit decision
      4. If window_size reached without confident decision -> emit best guess

    Returns: (class_id, class_name, confidence, num_packets_used)
    """

    def __init__(self,
                 model: torch.nn.Module,
                 confidence_threshold: float = 0.90,
                 window_size: int = DEFAULT_WINDOW,
                 min_packets: int = 3,
                 device: torch.device = None):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.graph = IncrementalFlowGraph(
            window_size=window_size,
            min_packets_before_classify=min_packets
        )

        self._best_pred   = None
        self._best_conf   = 0.0
        self._best_n_pkts = 0

    def ingest(self, packet_vector: np.ndarray,
               timestamp: Optional[float] = None
               ) -> Optional[Tuple[int, str, float, int]]:
        """
        Process one packet.

        Returns None if no confident decision yet.
        Returns (class_id, class_name, confidence, num_packets_used)
        when confidence threshold is exceeded or window is full.
        """
        self.graph.add_packet(packet_vector, timestamp)

        if not self.graph.is_ready():
            return None

        data = self.graph.to_pyg()
        if data is None:
            return None

        data = data.to(self.device)

        # Add dummy batch vector for single-graph inference
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long,
                                 device=self.device)

        with torch.no_grad():
            logits = self.model(data)
            probs  = torch.softmax(logits, dim=1)[0]  # [num_classes]

        conf       = float(probs.max().item())
        pred_class = int(probs.argmax().item())
        n_pkts     = self.graph.num_packets()

        # Track best prediction so far
        if conf > self._best_conf:
            self._best_conf   = conf
            self._best_pred   = pred_class
            self._best_n_pkts = n_pkts

        # Early exit if confident enough
        if conf >= self.confidence_threshold:
            return (pred_class, CLASS_NAMES.get(pred_class, str(pred_class)),
                    conf, n_pkts)

        # Force decision when window is full
        if n_pkts >= self.graph.window_size:
            return (self._best_pred,
                    CLASS_NAMES.get(self._best_pred, str(self._best_pred)),
                    self._best_conf, self._best_n_pkts)

        return None  # still collecting packets

    def reset(self):
        """Reset for a new flow."""
        self.graph.reset()
        self._best_pred   = None
        self._best_conf   = 0.0
        self._best_n_pkts = 0


# ============================================================
# Streaming benchmark utility
# ============================================================

def benchmark_early_exit(model, test_dataset, device,
                          confidence_threshold=0.90):
    """
    Simulate streaming classification on the test dataset.
    Reports:
      - accuracy
      - average packets used before decision
      - average latency reduction vs waiting for full window
    """
    model.eval()
    correct = 0
    total   = 0
    total_pkts_used  = 0
    max_window       = DEFAULT_WINDOW

    for graph_data in test_dataset:
        true_label = int(graph_data.y.item())
        x = graph_data.x.numpy()   # [num_nodes, 1500]

        classifier = EarlyExitClassifier(
            model=model,
            confidence_threshold=confidence_threshold,
            device=device
        )

        result = None
        for pkt_vec in x:
            result = classifier.ingest(pkt_vec)
            if result is not None:
                break

        if result is None:
            # Force decision
            data = classifier.graph.to_pyg()
            if data is not None:
                data = data.to(device)
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long,
                                         device=device)
                with torch.no_grad():
                    logits = model(data)
                    pred   = int(logits.argmax(dim=1).item())
                    conf   = float(torch.softmax(logits, dim=1).max().item())
                result = (pred, CLASS_NAMES.get(pred), conf,
                          classifier.graph.num_packets())

        if result:
            pred_class, _, conf, pkts_used = result
            correct += int(pred_class == true_label)
            total_pkts_used += pkts_used
        total += 1

    acc              = correct / max(total, 1)
    avg_pkts         = total_pkts_used / max(total, 1)
    latency_reduction = (max_window - avg_pkts) / max_window * 100

    print(f"\nStreaming Early-Exit Results (threshold={confidence_threshold}):")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Avg packets used  : {avg_pkts:.2f} / {max_window}")
    print(f"  Latency reduction : {latency_reduction:.1f}%")

    return acc, avg_pkts, latency_reduction


# -----------------------------------
# Quick test
# -----------------------------------
if __name__ == "__main__":
    from model import TrafficGNN
    from graph_builder import flow_to_graph

    np.random.seed(42)
    model = TrafficGNN(input_dim=1500, hidden_dim=64, num_classes=4)
    device = torch.device("cpu")

    # Simulate a streaming flow of 10 packets
    print("=== Incremental graph test ===")
    inc = IncrementalFlowGraph(window_size=10, min_packets_before_classify=3)
    for i in range(10):
        pkt = np.random.rand(1500).astype(np.float32)
        inc.add_packet(pkt)
        g = inc.to_pyg()
        if g is not None:
            print(f"  After pkt {i+1}: nodes={g.num_nodes} "
                  f"edges={g.num_edges}")

    print("\n=== Early exit test ===")
    clf = EarlyExitClassifier(model, confidence_threshold=0.90, device=device)
    for i in range(10):
        pkt = np.random.rand(1500).astype(np.float32)
        result = clf.ingest(pkt)
        if result:
            cls_id, cls_name, conf, n = result
            print(f"  Decision at packet {i+1}: "
                  f"class={cls_name} conf={conf:.3f} "
                  f"pkts_used={n}")
            break
    else:
        print("  No confident early exit within 10 packets (expected with random data)")

    print("\n=== Benchmark test ===")
    dataset = []
    for lbl in range(4):
        for _ in range(10):
            flow = np.random.rand(8, 1500).astype(np.float32)
            g = flow_to_graph(flow, lbl)
            if g is not None:
                dataset.append(g)

    benchmark_early_exit(model, dataset, device, confidence_threshold=0.90)
