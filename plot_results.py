import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ["chat", "email", "streaming", "voip"]
COLORS = {"base": "#6E96A8", "n1": "#1F5C4C", "n2": "#2E7D5B", "n3": "#4F8F6B", "n5": "#7FA99B"}
LABELS = {"base": "Base", "n1": "N1\nAdaptive", "n2": "N2\nHierarchical", "n3": "N3\nTemporal", "n5": "N5\nExplainable"}


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved graph: {path}")


def plot_model_comparison(results, out_dir="plots"):
    keys = [k for k in ["base", "n1", "n2", "n3", "n5"] if k in results]
    if not keys:
        return
    has_n4 = "n4" in results
    fig, axes = plt.subplots(1, 2 if has_n4 else 1, figsize=(13 if has_n4 else 7.5, 4.8))
    ax = axes[0] if has_n4 else axes
    accs = [results[k]["acc"][0] for k in keys]
    errs = [results[k]["acc"][1] for k in keys]
    bars = ax.bar([LABELS[k] for k in keys], accs, yerr=errs, capsize=5, color=[COLORS[k] for k in keys])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Accuracy (5-fold average)")
    ax.set_title("Base model vs. novelties")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.3f}", ha="center", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    if has_n4:
        n4 = results["n4"]
        ax2 = axes[1]
        vals = [n4["ssl_acc"], n4["baseline_acc"]]
        b2 = ax2.bar([f"With SSL pre-training\n({n4['n_labeled']} labels)", f"Without pre-training\n(same {n4['n_labeled']} labels)"],
                     vals, color=["#1F5C4C", "#9AA5AC"], width=0.5)
        ax2.set_ylim(0, 1.08); ax2.set_ylabel("Accuracy"); ax2.set_title("N4: self-supervised pre-training")
        for b, a in zip(b2, vals):
            ax2.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.3f}", ha="center", fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir, "1_model_comparison.png")


def plot_confusion(results, out_dir="plots"):
    c = results.get("base_confusion")
    if not c:
        return
    cm = confusion_matrix(c["y_true"], c["y_pred"], labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
    ax.set_title("Base model confusion matrix (last fold)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, out_dir, "2_confusion_matrix.png")


def plot_early_exit(results, out_dir="plots"):
    n6 = results.get("n6")
    if not n6:
        return
    th = sorted(n6.keys())
    acc = [n6[t]["acc"] * 100 for t in th]
    lat = [n6[t]["latency_reduction"] for t in th]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(th, acc, "o-", color="#1F5C4C", label="Accuracy (%)")
    ax.plot(th, lat, "s--", color="#B8862A", label="Decision time saved (%)")
    for t, a, l in zip(th, acc, lat):
        ax.text(t, a + 1.5, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(t, l + 1.5, f"{l:.1f}", ha="center", fontsize=9)
    ax.set_xlabel("Confidence needed before deciding"); ax.set_ylabel("Percent")
    ax.set_ylim(0, 105); ax.set_title("N6 early exit: accuracy vs. time saved")
    ax.legend(loc="center right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir, "3_early_exit_tradeoff.png")


def plot_graph_examples(dataset_path="data/nonvpn_graph_dataset.pt", out_dir="plots"):
    """Draws one real flow graph per class: dots = packets, solid line = time
    order, dashed blue line = similarity edge."""
    if not os.path.exists(dataset_path):
        return
    ds = torch.load(dataset_path, weights_only=False)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for lab, ax in enumerate(axes):
        g = next((d for d in ds if int(d.y.item()) == lab), None)
        ax.set_axis_off()
        if g is None:
            continue
        n = g.num_nodes
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = np.stack([np.cos(ang), np.sin(ang)], 1)
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in g.edge_index.t().tolist() if a != b}
        for a, b in pairs:
            seq = abs(a - b) == 1
            ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                    color="black" if seq else "#3B6FD4", ls="-" if seq else "--", lw=1.6 if seq else 1.1, zorder=1)
        ax.scatter(pos[:, 0], pos[:, 1], s=420, color="#1F5C4C", zorder=2)
        for i in range(n):
            ax.text(pos[i, 0], pos[i, 1], str(i), color="white", ha="center", va="center", fontweight="bold", zorder=3)
        ax.set_title(f"{CLASS_NAMES[lab]}\n{n} packets, {len(pairs)} edges")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    fig.suptitle("Flow graphs built by the code  (solid = time order, dashed = similarity)", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir, "4_graph_examples.png")


def plot_all(results=None, out_dir="plots"):
    print("\nGenerating output graphs ...")
    if results is None:
        p = "data/novelty_results.pt"
        results = torch.load(p, weights_only=False) if os.path.exists(p) else {}
    plot_model_comparison(results, out_dir)
    plot_confusion(results, out_dir)
    plot_early_exit(results, out_dir)
    plot_graph_examples(out_dir=out_dir)
    print(f"All graphs are in the '{out_dir}/' folder.")


if __name__ == "__main__":
    plot_all()
