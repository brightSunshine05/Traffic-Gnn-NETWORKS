import os
import torch
from tqdm import tqdm
from preprocessing import extract_flows_from_pcap, flows_to_vectors
from graph_builder import flow_to_graph
from temporal_edges import flow_to_temporal_graph
from adaptive_graph_builder import flow_to_candidate_graph
import synthetic_data

LABEL_MAP = {
    "chat":      0,
    "email":     1,
    "streaming": 2,
    "voip":      3,
}

MIN_NODES = 2


def _iter_flows_from_pcaps(data_path, max_packets):
    """Yields (label, flow_vectors, flow_times, flow_sizes) from real pcaps."""
    class_folders = sorted([
        f for f in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, f)) and f in LABEL_MAP
    ])
    for class_name in class_folders:
        label      = LABEL_MAP[class_name]
        class_path = os.path.join(data_path, class_name)
        pcap_files = sorted([
            f for f in os.listdir(class_path)
            if f.endswith((".pcap", ".pcapng"))
        ])
        print(f"\n[{class_name}] label={label} | files={len(pcap_files)}")
        for fname in tqdm(pcap_files, desc=f"  {class_name}"):
            file_path = os.path.join(class_path, fname)
            flows = extract_flows_from_pcap(file_path, max_packets=max_packets)
            flow_vecs, flow_times, flow_sizes = flows_to_vectors(flows)
            for vecs, times, sizes in zip(flow_vecs, flow_times, flow_sizes):
                yield label, vecs, times, sizes


def _iter_flows_from_synthetic(n_flows_per_class, max_packets, seed=0):
    """
    MODIFICATION (CHANGES.md item 4): yields flows from synthetic_data.py in
    the same (label, vecs, times, sizes) shape as _iter_flows_from_pcaps, so
    every graph-building function below is unaware of the data's origin.
    """
    data = synthetic_data.generate_synthetic_flows(
        n_flows_per_class=n_flows_per_class, max_packets=max_packets, seed=seed
    )
    for class_name, flows in data.items():
        label = LABEL_MAP[class_name]
        print(f"\n[{class_name}] label={label} | synthetic flows={len(flows)}")
        for vecs, times, sizes in flows:
            yield label, vecs, times, sizes


def build_graph_dataset(data_path="data/nonvpn", max_packets=10,
                         use_synthetic=None, n_flows_per_class=50, seed=0):
    """
    Builds the base, temporal, and adaptive-candidate datasets in one pass.

    use_synthetic: None -> auto-detect (use synthetic data if data_path has
    no class subfolders with pcap files); True/False force one path.

    Returns:
        base_dataset     : list of PyG Data (seq + similarity edges, weight=[E,1])
        temporal_dataset : list of PyG Data (+ temporal edge attrs, weight=[E,4])
        adaptive_dataset : list of PyG Data (candidate edges + flow_stats, Novelty 1)
    """
    if use_synthetic is None:
        use_synthetic = not (
            os.path.isdir(data_path)
            and any(
                os.path.isdir(os.path.join(data_path, c))
                and any(f.endswith((".pcap", ".pcapng"))
                        for f in os.listdir(os.path.join(data_path, c)))
                for c in LABEL_MAP if os.path.isdir(os.path.join(data_path, c))
            )
        )

    if use_synthetic:
        print(f"No real pcap captures found under '{data_path}' -- "
              f"falling back to synthetic_data.py (see CHANGES.md item 4).")
        flow_iter = _iter_flows_from_synthetic(n_flows_per_class, max_packets, seed)
    else:
        print(f"Reading real pcap captures from '{data_path}'.")
        flow_iter = _iter_flows_from_pcaps(data_path, max_packets)

    base_dataset, temporal_dataset, adaptive_dataset = [], [], []
    counts = {c: 0 for c in LABEL_MAP}

    for label, vecs, times, sizes in flow_iter:
        if len(vecs) < MIN_NODES:
            continue

        g_base = flow_to_graph(vecs, label)
        if g_base is not None:
            base_dataset.append(g_base)

        g_temp = flow_to_temporal_graph(vecs, times, sizes, label)
        if g_temp is not None:
            temporal_dataset.append(g_temp)

        g_adapt = flow_to_candidate_graph(vecs, label)
        if g_adapt is not None:
            adaptive_dataset.append(g_adapt)

        counts[[k for k, v in LABEL_MAP.items() if v == label][0]] += 1

    print("\nFlows built per class:", counts)
    return base_dataset, temporal_dataset, adaptive_dataset


if __name__ == "__main__":
    base_ds, temp_ds, adapt_ds = build_graph_dataset()

    print(f"\nTotal base graphs     : {len(base_ds)}")
    print(f"Total temporal graphs : {len(temp_ds)}")
    print(f"Total adaptive graphs : {len(adapt_ds)}")

    if base_ds:
        g = base_ds[0]
        print(f"\nBase graph example:")
        print(f"  nodes={g.num_nodes}  edges={g.num_edges}")
        print(f"  x={g.x.shape}  edge_attr={g.edge_attr.shape}  y={g.y}")

    if temp_ds:
        g = temp_ds[0]
        print(f"\nTemporal graph example:")
        print(f"  nodes={g.num_nodes}  edges={g.num_edges}")
        print(f"  x={g.x.shape}  edge_attr={g.edge_attr.shape}  y={g.y}")
        print(f"  edge_attr[0] = {g.edge_attr[0]}  (sim, iat, byterate, iat_norm)")

    if adapt_ds:
        g = adapt_ds[0]
        print(f"\nAdaptive candidate graph example:")
        print(f"  nodes={g.num_nodes}  edges={g.num_edges}  flow_stats={g.flow_stats.shape}")

    os.makedirs("data", exist_ok=True)
    torch.save(base_ds,   "data/nonvpn_graph_dataset.pt")
    torch.save(temp_ds,   "data/nonvpn_temporal_dataset.pt")
    torch.save(adapt_ds,  "data/nonvpn_adaptive_dataset.pt")
    print("\nSaved:")
    print("  data/nonvpn_graph_dataset.pt")
    print("  data/nonvpn_temporal_dataset.pt")
    print("  data/nonvpn_adaptive_dataset.pt")
