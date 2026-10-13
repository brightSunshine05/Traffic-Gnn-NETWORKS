import os
import torch
from tqdm import tqdm
from preprocessing import extract_flows_from_pcap, flows_to_vectors
from hierarchical_graph import flow_to_hierarchical_graph
import synthetic_data

LABEL_MAP = {
    "chat":      0,
    "email":     1,
    "streaming": 2,
    "voip":      3,
}

MIN_NODES = 2   # minimum packets per flow


def _has_real_pcaps(data_path):
    return (
        os.path.isdir(data_path)
        and any(
            os.path.isdir(os.path.join(data_path, c))
            and any(f.endswith((".pcap", ".pcapng"))
                    for f in os.listdir(os.path.join(data_path, c)))
            for c in LABEL_MAP if os.path.isdir(os.path.join(data_path, c))
        )
    )


def build_hierarchical_dataset(data_path="data/nonvpn", max_packets=10,
                                use_synthetic=None, n_flows_per_class=50, seed=0):
    if use_synthetic is None:
        use_synthetic = not _has_real_pcaps(data_path)

    dataset = []
    counts = {c: 0 for c in LABEL_MAP}

    if use_synthetic:
        print(f"No real pcap captures found under '{data_path}' -- "
              f"falling back to synthetic_data.py (see CHANGES.md item 4).")
        data = synthetic_data.generate_synthetic_flows(
            n_flows_per_class=n_flows_per_class, max_packets=max_packets, seed=seed
        )
        for class_name, flows in data.items():
            label = LABEL_MAP[class_name]
            print(f"\n[{class_name}] label={label} | synthetic flows={len(flows)}")
            for vecs, _times, _sizes in flows:
                if len(vecs) < MIN_NODES:
                    continue
                g = flow_to_hierarchical_graph(vecs, label)
                if g is not None:
                    dataset.append(g)
                    counts[class_name] += 1
        print("\nHierarchical graphs per class:", counts)
        return dataset

    class_folders = sorted([
        f for f in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, f)) and f in LABEL_MAP
    ])
    print("Classes:", class_folders)

    for class_name in class_folders:
        label      = LABEL_MAP[class_name]
        class_path = os.path.join(data_path, class_name)
        pcap_files = sorted([
            f for f in os.listdir(class_path)
            if f.endswith((".pcap", ".pcapng"))
        ])
        print(f"\n[{class_name}] label={label} | files={len(pcap_files)}")
        count = 0

        for fname in tqdm(pcap_files, desc=f"  {class_name}"):
            file_path = os.path.join(class_path, fname)
            flows = extract_flows_from_pcap(file_path, max_packets=max_packets)
            flow_vecs, _, _ = flows_to_vectors(flows)  # only need vectors here

            for vecs in flow_vecs:
                if len(vecs) < MIN_NODES:
                    continue
                g = flow_to_hierarchical_graph(vecs, label)
                if g is not None:
                    dataset.append(g)
                    count += 1

        print(f"  Hierarchical graphs: {count}")

    return dataset


if __name__ == "__main__":
    dataset = build_hierarchical_dataset()

    print(f"\nTotal hierarchical graphs: {len(dataset)}")
    if dataset:
        g = dataset[0]
        print(f"\nExample hierarchical graph:")
        print(f"  total nodes    : {g.num_nodes}")
        print(f"  n-gram nodes   : {g.num_ngram_nodes}")
        print(f"  packet nodes   : {g.num_packet_nodes}")
        print(f"  edges          : {g.num_edges}")
        print(f"  x shape        : {g.x.shape}")
        print(f"  edge_attr shape: {g.edge_attr.shape}")
        print(f"  node_type      : {g.node_type.tolist()[:12]}...")

    os.makedirs("data", exist_ok=True)
    save_path = "data/nonvpn_hierarchical_dataset.pt"
    torch.save(dataset, save_path)
    print(f"\nSaved: {save_path}")