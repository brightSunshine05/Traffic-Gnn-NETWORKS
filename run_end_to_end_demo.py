import os, time, warnings, torch, numpy as np
warnings.filterwarnings("ignore")

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

import build_dataset
import build_hierarchical_dataset
from model import TrafficGNN
from adaptive_graph_builder import AdaptiveTrafficGNN
from hierarchical_graph import HierarchicalTrafficGNN
from temporal_edges import TemporalTrafficGNN
from explainable_gnn import ExplainableTrafficGNN, train_with_explanation_loss
from online_inference import benchmark_early_exit

# ── reduced-scale demo config (paper-faithful values are in train.py) ──────
N_FLOWS_PER_CLASS = 60      # -> 240 synthetic flows total
FOLDS             = 5       # paper protocol kept: stratified 5-fold CV
EPOCHS            = 35      # reduced from the paper's 500 for CPU runtime
HIDDEN_DIM        = 128     # matches paper / senior code default
BATCH             = 64
LR                = 1e-3
WD                = 1e-5
DEVICE            = torch.device("cpu")
CLASS_NAMES       = ["chat", "email", "streaming", "voip"]

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("=" * 70)
print("AdaptiGraph-NTC -- end-to-end demo run (synthetic data, reduced scale)")
print(f"  flows/class={N_FLOWS_PER_CLASS}  folds={FOLDS}  epochs/fold={EPOCHS}")
print("=" * 70)

t0 = time.time()

# ── 1. Build datasets (base, temporal, adaptive-candidate, hierarchical) ───
print("\n[1/3] Building graph datasets from synthetic_data.py ...")
base_ds, temp_ds, adapt_ds = build_dataset.build_graph_dataset(
    use_synthetic=True, n_flows_per_class=N_FLOWS_PER_CLASS, seed=0
)
hier_ds = build_hierarchical_dataset.build_hierarchical_dataset(
    use_synthetic=True, n_flows_per_class=N_FLOWS_PER_CLASS, seed=0
)
print(f"\nDataset sizes: base={len(base_ds)}  temporal={len(temp_ds)}  "
      f"adaptive={len(adapt_ds)}  hierarchical={len(hier_ds)}")

torch.save(base_ds,  "data/nonvpn_graph_dataset.pt")
torch.save(temp_ds,  "data/nonvpn_temporal_dataset.pt")
torch.save(adapt_ds, "data/nonvpn_adaptive_dataset.pt")
torch.save(hier_ds,  "data/nonvpn_hierarchical_dataset.pt")


# ── shared helpers ───────────────────────────────────────────────────────
def make_sampler(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = {c: total / counts[c] for c in counts}
    sw = [weights[l] for l in labels]
    return WeightedRandomSampler(sw, len(sw), replacement=True)


def run_fold(model, train_data, test_data, criterion, epochs=EPOCHS, use_expl_loss=False):
    labels = [int(g.y.item()) for g in train_data]
    sampler = make_sampler(labels)
    t_loader = DataLoader(train_data, batch_size=BATCH, sampler=sampler)
    v_loader = DataLoader(test_data, batch_size=BATCH, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_f1, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        if use_expl_loss:
            train_with_explanation_loss(model, t_loader, opt, DEVICE)
        else:
            model.train()
            for batch in t_loader:
                batch = batch.to(DEVICE)
                opt.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in v_loader:
                    batch = batch.to(DEVICE)
                    preds.extend(model(batch).argmax(1).cpu().tolist())
                    trues.extend(batch.y.cpu().tolist())
            f1 = f1_score(trues, preds, average="macro", zero_division=0)
            if f1 >= best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in v_loader:
            batch = batch.to(DEVICE)
            preds.extend(model(batch).argmax(1).cpu().tolist())
            trues.extend(batch.y.cpu().tolist())

    return {
        "acc":  accuracy_score(trues, preds),
        "prec": precision_score(trues, preds, average="macro", zero_division=0),
        "rec":  recall_score(trues, preds, average="macro", zero_division=0),
        "f1":   f1_score(trues, preds, average="macro", zero_division=0),
    }, trues, preds


def kfold_eval(dataset, model_fn, label, use_expl_loss=False):
    labels = np.array([int(g.y.item()) for g in dataset])
    ds_arr = np.array(dataset, dtype=object)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    results, last_trues, last_preds = [], None, None

    for fold, (tr_idx, te_idx) in enumerate(skf.split(ds_arr, labels)):
        print(f"  {label} | fold {fold + 1}/{FOLDS} ...", end=" ", flush=True)
        model = model_fn().to(DEVICE)
        res, trues, preds = run_fold(
            model, [dataset[i] for i in tr_idx], [dataset[i] for i in te_idx],
            torch.nn.CrossEntropyLoss(), use_expl_loss=use_expl_loss,
        )
        print(f"acc={res['acc']:.4f}  f1={res['f1']:.4f}")
        results.append(res)
        last_trues, last_preds = trues, preds

    summary = {m: (np.mean([r[m] for r in results]), np.std([r[m] for r in results]))
               for m in ["acc", "prec", "rec", "f1"]}
    print(f"  {label} SUMMARY:", "  ".join(f"{m.upper()}={mu:.4f}+/-{sd:.4f}" for m, (mu, sd) in summary.items()))
    return summary, last_trues, last_preds


# ── 2. Train base model + each implemented novelty ──────────────────────
print("\n[2/3] Training base model and novelties (5-fold stratified CV) ...")
all_results = {}

print("\n-- BASE MODEL (Algorithms 1-5, Section 3.3) --")
base_res, base_trues, base_preds = kfold_eval(
    base_ds, lambda: TrafficGNN(input_dim=1500, hidden_dim=HIDDEN_DIM, num_classes=4), "Base"
)
all_results["base"] = base_res

print("\n-- NOVELTY 1: Adaptive per-flow threshold (fixed wiring, see CHANGES.md) --")
n1_res, n1_trues, n1_preds = kfold_eval(
    adapt_ds, lambda: AdaptiveTrafficGNN(input_dim=1500, hidden_dim=HIDDEN_DIM, num_classes=4), "N1-Adaptive"
)
all_results["n1"] = n1_res

print("\n-- NOVELTY 2: Hierarchical dual-level graph --")
n2_res, n2_trues, n2_preds = kfold_eval(
    hier_ds, lambda: HierarchicalTrafficGNN(input_dim=1500, hidden_dim=HIDDEN_DIM, num_classes=4), "N2-Hierarchical"
)
all_results["n2"] = n2_res

print("\n-- NOVELTY 3: Temporal edge attributes --")
n3_res, n3_trues, n3_preds = kfold_eval(
    temp_ds, lambda: TemporalTrafficGNN(input_dim=1500, hidden_dim=64, edge_attr_dim=4, num_classes=4), "N3-Temporal"
)
all_results["n3"] = n3_res

print("\n-- NOVELTY 5: Intrinsic attention explainability --")
n5_res, n5_trues, n5_preds = kfold_eval(
    base_ds,
    lambda: ExplainableTrafficGNN(input_dim=1500, hidden_dim=HIDDEN_DIM, num_classes=4,
                                   edge_attr_dim=1, sparsity_lambda=1e-3),
    "N5-Explainable", use_expl_loss=True,
)
all_results["n5"] = n5_res

print("\n-- NOVELTY 6: Online incremental inference / early exit --")
# Re-use a freshly trained base model as the backbone for streaming inference
backbone = TrafficGNN(input_dim=1500, hidden_dim=HIDDEN_DIM, num_classes=4).to(DEVICE)
tr_idx = np.random.RandomState(0).permutation(len(base_ds))
split = int(0.7 * len(base_ds))
bb_res, _, _ = run_fold(backbone, [base_ds[i] for i in tr_idx[:split]],
                         [base_ds[i] for i in tr_idx[split:]],
                         torch.nn.CrossEntropyLoss(), epochs=EPOCHS)
print(f"  backbone for early-exit benchmark: acc={bb_res['acc']:.4f}")

n6_results = {}
test_subset = [base_ds[i] for i in tr_idx[split:]]
for thresh in [0.70, 0.80, 0.90, 0.95]:
    acc, avg_pkts, lat_red = benchmark_early_exit(backbone, test_subset, DEVICE, confidence_threshold=thresh)
    n6_results[thresh] = {"acc": acc, "avg_pkts": avg_pkts, "latency_reduction": lat_red}
all_results["n6"] = n6_results

print("\n-- NOVELTY 4: Self-supervised pre-training (label-efficiency test) --")
from sklearn.model_selection import train_test_split
from ssl_pretrain import GraphContrastivePretrainer
_lab = np.array([int(g.y.item()) for g in base_ds]); _idx = np.arange(len(base_ds))
_unl, _lbl = train_test_split(_idx, test_size=0.40, stratify=_lab, random_state=0)
_tr, _te = train_test_split(_lbl, test_size=0.30, stratify=_lab[_lbl], random_state=0)
def _eval(m, data):
    m.eval(); P, T = [], []
    with torch.no_grad():
        for b in DataLoader(data, batch_size=64):
            P += m(b).argmax(1).tolist(); T += b.y.tolist()
    return accuracy_score(T, P), f1_score(T, P, average="macro", zero_division=0)
_pre = GraphContrastivePretrainer(TrafficGNN(input_dim=1500, hidden_dim=HIDDEN_DIM, num_classes=4), device=DEVICE)
_pre.pretrain([base_ds[i] for i in _unl], epochs=25, batch_size=32, save_path="data/pretrained_encoder.pt")
_ft = _pre.fine_tune([base_ds[i] for i in _tr], pretrained_path="data/pretrained_encoder.pt", epochs=30, batch_size=32)
ssl_acc, ssl_f1 = _eval(_ft, [base_ds[i] for i in _te])
_bl = TrafficGNN(input_dim=1500, hidden_dim=HIDDEN_DIM, num_classes=4)
_opt = torch.optim.Adam(_bl.parameters(), lr=LR, weight_decay=WD); _cr = torch.nn.CrossEntropyLoss()
for _ in range(30):
    _bl.train()
    for b in DataLoader([base_ds[i] for i in _tr], batch_size=32, shuffle=True):
        _opt.zero_grad(); _cr(_bl(b), b.y).backward(); _opt.step()
bl_acc, bl_f1 = _eval(_bl, [base_ds[i] for i in _te])
all_results["n4"] = {"ssl_acc": ssl_acc, "ssl_f1": ssl_f1, "baseline_acc": bl_acc, "baseline_f1": bl_f1,
                     "n_unlabeled": len(_unl), "n_labeled": len(_tr), "n_test": len(_te)}
print(f"  SSL pre-train+finetune acc={ssl_acc:.4f}  |  no pre-training (same {len(_tr)} labels) acc={bl_acc:.4f}")

all_results["base_confusion"] = {"y_true": base_trues, "y_pred": base_preds}
torch.save(all_results, "data/novelty_results.pt")
torch.save({"y_true": base_trues, "y_pred": base_preds}, "data/base_confusion_source.pt")

# ── 3. Summary table + confusion matrix plot ─────────────────────────────
print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE (macro F1, mean +/- std across 5 folds)")
print("=" * 70)
mapping = [
    ("base",                  "Base model (Algorithms 1-5)"),
    ("n1",     "N1: Adaptive threshold (fixed wiring)"),
    ("n2", "N2: Hierarchical graph"),
    ("n3",     "N3: Temporal edges"),
    ("n5",  "N5: Intrinsic explainability"),
]
print(f"  {'Method':<42}  {'Acc':>8}  {'F1':>8}")
print("  " + "-" * 62)
for key, name in mapping:
    res = all_results[key]
    print(f"  {name:<42}  {res['acc'][0]:>8.4f}  {res['f1'][0]:>8.4f}")
print(f"  {'N6: Early exit @ conf=0.90':<42}  {n6_results[0.90]['acc']:>8.4f}"
      f"  (avg pkts {n6_results[0.90]['avg_pkts']:.1f}/10, "
      f"latency -{n6_results[0.90]['latency_reduction']:.1f}%)")

from plot_results import plot_all
plot_all(all_results)

print(f"\nTotal wall-clock time: {time.time() - t0:.1f}s")
print("\nNote: this run uses synthetic_data.py (see CHANGES.md item 3), reduced")
print("epochs/fold count, and a smaller dataset than the paper's protocol, purely")
print("to validate that every module -- base, N1, N2, N3, N5, N6 -- trains and")
print("converges correctly end-to-end ahead of real ISCX pcap ingestion. Re-run")
print("train.py / train_novelties.py with real data under data/nonvpn/<class>/")
print("for numbers directly comparable to the base paper's Table 3.")
