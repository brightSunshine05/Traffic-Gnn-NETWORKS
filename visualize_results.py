import os, torch, numpy as np, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("plots", exist_ok=True)

# ── colour palette (colorblind-safe) ────────────────────────────────────────
C = {
    "base":    "#4878CF",
    "n1":      "#6ACC65",
    "n2":      "#D65F5F",
    "n3":      "#B47CC7",
    "n4":      "#C4AD66",
    "n5":      "#77BEDB",
    "n6":      "#F28522",
    "chat":    "#4878CF",
    "email":   "#6ACC65",
    "stream":  "#D65F5F",
    "voip":    "#B47CC7",
}
CLASS_NAMES = ["chat", "email", "streaming", "voip"]

def save(name):
    path = f"plots/{name}"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 1. Class distribution (dataset imbalance)
# ════════════════════════════════════════════════════════════════════════════
def plot_class_distribution():
    try:
        ds = torch.load("data/nonvpn_graph_dataset.pt", weights_only=False)
    except FileNotFoundError:
        print("  Skipping class_distribution.png -- dataset not found"); return

    from collections import Counter
    counts = Counter([int(g.y.item()) for g in ds])
    labels = [CLASS_NAMES[i] for i in sorted(counts)]
    values = [counts[i] for i in sorted(counts)]
    colors = [C["chat"], C["email"], C["stream"], C["voip"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Number of flow graphs", fontsize=11)
    ax.set_title("Dataset class distribution (Non-VPN)", fontsize=12)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(v), ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save("class_distribution.png")

def plot_training_curves():
    log_path = "data/training_log.pt"
    if not os.path.exists(log_path):
        print("  Skipping base_training_curves.png -- no training_log.pt")
        print("  (Add torch.save({'loss':losses,'acc':accs}, 'data/training_log.pt')")
        print("   to train.py if you want this plot)")
        return

    log = torch.load(log_path, weights_only=False)
    epochs = list(range(1, len(log["loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, log["loss"], color=C["base"], lw=1.5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training loss"); ax1.grid(alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    ax2.plot(epochs, log["acc"], color=C["n1"], lw=1.5, label="Test acc")
    ax2.axhline(np.max(log["acc"]), color="gray", ls="--", lw=0.8,
                label=f"Best {np.max(log['acc']):.4f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Test accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("Base model training (fold 1)", fontsize=12)
    fig.tight_layout()
    save("base_training_curves.png")


# ════════════════════════════════════════════════════════════════════════════
# 3. Confusion matrix (base paper, fold 5 from screenshot data)
# ════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix():
    # Values from the screenshot you shared (fold 5 best)
    cm = np.array([
        [0.8421, 0.0421, 0.0421, 0.0737],
        [0.0084, 0.9704, 0.0000, 0.0212],
        [0.2333, 0.1333, 0.6333, 0.0000],
        [0.0029, 0.0000, 0.0000, 0.9971],
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix -- base model (fold 5)", fontsize=11)

    for i in range(4):
        for j in range(4):
            val   = cm[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save("base_confusion_matrix.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. Novelty comparison bar chart (F1 scores)
# ════════════════════════════════════════════════════════════════════════════
def plot_novelty_comparison():
    res_path = "data/novelty_results.pt"
    if not os.path.exists(res_path):
        print("  Skipping novelty_comparison.png -- run train_novelties.py first")
        return

    results = torch.load(res_path, weights_only=False)

    # Base paper result (from screenshot: cross-val F1 mean ~0.82)
    entries = [
        ("Base paper\n(reproduced)", 0.8220, 0.0183, C["base"]),
    ]
    mapping = [
        ("novelty1_adaptive",    "N1: Adaptive\nthreshold",    C["n1"]),
        ("novelty2_hierarchical","N2: Hierarchical\ngraph",     C["n2"]),
        ("novelty3_temporal",    "N3: Temporal\nedges",         C["n3"]),
        ("novelty4_ssl",         "N4: SSL\npre-training",       C["n4"]),
        ("novelty5_explainable", "N5: Intrinsic\nexplainability",C["n5"]),
    ]
    for key, label, color in mapping:
        res = results.get(key)
        if res is not None:
            mu, sd = res["f1"]
            entries.append((label, mu, sd, color))

    labels  = [e[0] for e in entries]
    means   = [e[1] for e in entries]
    stds    = [e[2] for e in entries]
    colors  = [e[3] for e in entries]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.8,
                  error_kw={"elinewidth": 1.2, "ecolor": "gray"})
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Macro F1 score", fontsize=11)
    ax.set_title("Method comparison -- Non-VPN dataset (5-fold CV)", fontsize=12)
    ax.set_ylim(0.70, 1.02)
    ax.axhline(0.8220, color="gray", ls="--", lw=0.8, label="Base paper F1")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    for bar, mu in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{mu:.4f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save("novelty_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# 5. Threshold ablation: fixed vs adaptive (Novelty 1)
# ════════════════════════════════════════════════════════════════════════════
def plot_threshold_ablation():
    res_path = "data/novelty_results.pt"
    if not os.path.exists(res_path):
        print("  Skipping threshold_ablation.png"); return

    results = torch.load(res_path, weights_only=False)
    res_n1  = results.get("novelty1_adaptive")
    if res_n1 is None:
        print("  Skipping threshold_ablation.png -- novelty1 not in results"); return

    # Compare fixed percentile thresholds vs adaptive
    metrics   = ["acc", "prec", "rec", "f1"]
    fixed_mu  = [0.9636, 0.8220, 0.8322, 0.8188]   # from your screenshot
    fixed_sd  = [0.0025, 0.0183, 0.0118, 0.0118]
    adapt_mu  = [res_n1[m][0] for m in metrics]
    adapt_sd  = [res_n1[m][1] for m in metrics]

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, fixed_mu, w, yerr=fixed_sd, capsize=4,
           label="Fixed 90th-percentile", color=C["base"], alpha=0.85,
           error_kw={"elinewidth": 1.2})
    ax.bar(x + w/2, adapt_mu, w, yerr=adapt_sd, capsize=4,
           label="Adaptive threshold (N1)", color=C["n1"], alpha=0.85,
           error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Score"); ax.set_ylim(0.70, 1.02)
    ax.set_title("Novelty 1: Adaptive vs fixed similarity threshold", fontsize=11)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    save("threshold_ablation.png")


# ════════════════════════════════════════════════════════════════════════════
# 6. Temporal IAT distributions per class (Novelty 3 justification)
# Shows WHY timing helps -- different classes have distinct IAT patterns
# ════════════════════════════════════════════════════════════════════════════
def plot_temporal_iat():
    temp_path = "data/nonvpn_temporal_dataset.pt"
    if not os.path.exists(temp_path):
        print("  Skipping temporal_iat_boxplot.png -- temporal dataset not found"); return

    ds = torch.load(temp_path, weights_only=False)

    # Collect IAT values (edge_attr[:,1] = iat_log) per class
    iat_by_class = {i: [] for i in range(4)}
    for g in ds:
        lbl = int(g.y.item())
        if g.edge_attr is not None and g.edge_attr.shape[1] >= 2:
            # edge_attr: [E, 4] -> col 1 is iat_log
            iat_vals = g.edge_attr[:, 1].numpy().tolist()
            iat_by_class[lbl].extend(iat_vals)

    fig, ax = plt.subplots(figsize=(7, 4))
    data_to_plot = [iat_by_class[i] for i in range(4)]
    bp = ax.boxplot(data_to_plot, labels=CLASS_NAMES, patch_artist=True,
                    notch=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 2})

    class_colors = [C["chat"], C["email"], C["stream"], C["voip"]]
    for patch, color in zip(bp["boxes"], class_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Log-scaled inter-arrival time (normalised)", fontsize=10)
    ax.set_title("Novelty 3: IAT distribution per traffic class", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    note = ("VoIP shows tight IAT (regular cadence).\n"
            "Streaming is bursty (high variance).\n"
            "Email has long inter-burst silences.")
    ax.text(0.98, 0.97, note, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))
    fig.tight_layout()
    save("temporal_iat_boxplot.png")


# ════════════════════════════════════════════════════════════════════════════
# 7. Early-exit accuracy vs packets used (Novelty 6)
# ════════════════════════════════════════════════════════════════════════════
def plot_early_exit():
    res_path = "data/novelty_results.pt"
    if not os.path.exists(res_path):
        print("  Skipping early_exit_tradeoff.png"); return

    results = torch.load(res_path, weights_only=False)
    n6      = results.get("novelty6_early_exit")
    if n6 is None:
        print("  Skipping early_exit_tradeoff.png -- novelty6 not in results"); return

    thresholds = sorted(n6.keys())
    accs       = [n6[t]["acc"]               for t in thresholds]
    avg_pkts   = [n6[t]["avg_pkts"]          for t in thresholds]
    lat_reds   = [n6[t]["latency_reduction"] for t in thresholds]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, accs,     "o-", color=C["base"],  lw=2, label="Accuracy")
    ax2.plot(thresholds, lat_reds, "s--", color=C["n6"],   lw=2, label="Latency reduction (%)")

    ax1.set_xlabel("Confidence threshold", fontsize=11)
    ax1.set_ylabel("Classification accuracy", fontsize=11, color=C["base"])
    ax2.set_ylabel("Latency reduction (%)",   fontsize=11, color=C["n6"])
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

    ax1.set_title("Novelty 6: Early-exit accuracy vs latency reduction", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.spines[["top"]].set_visible(False)
    fig.tight_layout()
    save("early_exit_tradeoff.png")


# ════════════════════════════════════════════════════════════════════════════
# 8. Attention edge heatmap (Novelty 5) -- uses saved explanation
# ════════════════════════════════════════════════════════════════════════════
def plot_attention_heatmap():
    expl_path = "data/sample_explanation.pt"
    if not os.path.exists(expl_path):
        print("  Skipping attention_edge_heatmap.png -- run train_novelties.py first"); return

    expl   = torch.load(expl_path, weights_only=False)
    edges  = expl.get("top_edges", [])
    if not edges:
        print("  No edges in explanation"); return

    # Build a small attention matrix for the top edges
    node_ids = sorted(set([e[0] for e in edges] + [e[1] for e in edges]))
    n        = len(node_ids)
    id_map   = {v: i for i, v in enumerate(node_ids)}
    mat      = np.zeros((n, n))
    for src, dst, score in edges:
        if src in id_map and dst in id_map:
            mat[id_map[src], id_map[dst]] = score

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"p{v}" for v in node_ids], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"p{v}" for v in node_ids], fontsize=8)
    ax.set_xlabel("Destination packet node"); ax.set_ylabel("Source packet node")
    ax.set_title(f"Novelty 5: Edge attention scores\n"
                 f"(predicted: {expl.get('class_name','?')}, "
                 f"conf={expl.get('confidence',0):.3f})", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save("attention_edge_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
# Run all plots
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating plots ...\n")
    plot_class_distribution()
    plot_training_curves()
    plot_confusion_matrix()
    plot_novelty_comparison()
    plot_threshold_ablation()
    plot_temporal_iat()
    plot_early_exit()
    plot_attention_heatmap()

    print("\nAll plots saved to plots/")
    print("Files:")
    for f in sorted(os.listdir("plots")):
        print(f"  plots/{f}")