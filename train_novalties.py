import os, sys, time, torch, numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

# ── shared hyper-parameters (paper section 4.2.2) ──────────────────────────
BATCH  = 256
EPOCHS = 200          # 200 for novelty comparisons (500 for full runs)
LR     = 1e-3
WD     = 1e-5
FOLDS  = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

os.makedirs("data", exist_ok=True)

# ── helper: weighted sampler ────────────────────────────────────────────────
def make_sampler(labels):
    counts  = Counter(labels)
    total   = sum(counts.values())
    weights = {c: total / counts[c] for c in counts}
    sw      = [weights[l] for l in labels]
    return WeightedRandomSampler(sw, len(sw), replacement=True)

# ── helper: one fold train+eval ─────────────────────────────────────────────
def run_fold(model, train_data, test_data, criterion, epochs=EPOCHS):
    labels   = [int(g.y.item()) for g in train_data]
    sampler  = make_sampler(labels)
    t_loader = DataLoader(train_data, batch_size=BATCH, sampler=sampler)
    v_loader = DataLoader(test_data,  batch_size=BATCH, shuffle=False)
    opt      = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_f1, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for batch in t_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out  = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        if ep % 20 == 0 or ep == epochs:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in v_loader:
                    batch = batch.to(DEVICE)
                    out   = model(batch)
                    preds.extend(out.argmax(1).cpu().tolist())
                    trues.extend(batch.y.cpu().tolist())
            f1 = f1_score(trues, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1    = f1
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
    }

def kfold_eval(dataset, model_fn, label="novelty"):
    """5-fold stratified CV.  model_fn() must return a fresh model each call."""
    labels  = np.array([int(g.y.item()) for g in dataset])
    ds_arr  = np.array(dataset, dtype=object)
    skf     = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    results = []
    crit    = torch.nn.CrossEntropyLoss()

    for fold, (tr_idx, te_idx) in enumerate(skf.split(ds_arr, labels)):
        print(f"  {label} | fold {fold+1}/{FOLDS} ...", end=" ", flush=True)
        model = model_fn().to(DEVICE)
        res   = run_fold(model,
                         [dataset[i] for i in tr_idx],
                         [dataset[i] for i in te_idx],
                         crit)
        print(f"acc={res['acc']:.4f}  f1={res['f1']:.4f}")
        results.append(res)

    summary = {m: (np.mean([r[m] for r in results]),
                   np.std( [r[m] for r in results]))
               for m in ["acc","prec","rec","f1"]}
    print(f"  {label} SUMMARY:")
    for m,(mu,sd) in summary.items():
        print(f"    {m.upper():4s}: {mu:.4f} +/- {sd:.4f}")
    return summary

# ════════════════════════════════════════════════════════════════════════════
# Load datasets
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Loading datasets ...")
base_ds  = torch.load("data/nonvpn_graph_dataset.pt",       weights_only=False)
temp_ds  = torch.load("data/nonvpn_temporal_dataset.pt",    weights_only=False)
adapt_ds = torch.load("data/nonvpn_adaptive_dataset.pt",     weights_only=False)

hier_path = "data/nonvpn_hierarchical_dataset.pt"
hier_ds   = torch.load(hier_path, weights_only=False) if os.path.exists(hier_path) else None

print(f"  Base dataset     : {len(base_ds)}")
print(f"  Temporal dataset : {len(temp_ds)}")
print(f"  Adaptive dataset : {len(adapt_ds)}")
print(f"  Hierarchical     : {len(hier_ds) if hier_ds else 'not built yet'}")

all_results = {}

# ════════════════════════════════════════════════════════════════════════════
# Novelty 1: Adaptive Threshold
#
# MODIFICATION (CHANGES.md item 1): the original version of this section
# just re-trained the ordinary TrafficGNN on the same fixed-threshold
# base_ds graphs, so AdaptiveThresholdNet and soft_edge_mask() were never
# actually exercised during training -- the "adaptive" threshold was fully
# decorative. This version trains AdaptiveTrafficGNN (adaptive_graph_
# builder.py) on the candidate-edge dataset (adapt_ds), so the threshold
# MLP genuinely receives gradients from the classification loss and the
# learned per-flow theta genuinely changes which edges the GNN reads.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("NOVELTY 1: Adaptive Threshold (real end-to-end wiring)")
print("="*60)
from adaptive_graph_builder import AdaptiveTrafficGNN

def n1_model():
    return AdaptiveTrafficGNN(input_dim=1500, hidden_dim=128, num_classes=4)

n1_res = kfold_eval(adapt_ds, n1_model, label="Novelty1-AdaptiveThreshold")
all_results["novelty1_adaptive"] = n1_res

# ════════════════════════════════════════════════════════════════════════════
# Novelty 2: Hierarchical Graph
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("NOVELTY 2: Hierarchical Dual-Level Graph")
print("="*60)

if hier_ds is not None:
    from hierarchical_graph import HierarchicalTrafficGNN

    def n2_model():
        return HierarchicalTrafficGNN(input_dim=1500, hidden_dim=128, num_classes=4)

    n2_res = kfold_eval(hier_ds, n2_model, label="Novelty2-Hierarchical")
    all_results["novelty2_hierarchical"] = n2_res
else:
    print("  Skipped -- run build_hierarchical_dataset.py first")
    all_results["novelty2_hierarchical"] = None

# ════════════════════════════════════════════════════════════════════════════
# Novelty 3: Temporal Edge Attributes
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("NOVELTY 3: Temporal Edge Attributes")
print("="*60)
from temporal_edges import TemporalTrafficGNN

def n3_model():
    return TemporalTrafficGNN(input_dim=1500, hidden_dim=64,
                              edge_attr_dim=4, num_classes=4)

n3_res = kfold_eval(temp_ds, n3_model, label="Novelty3-TemporalEdges")
all_results["novelty3_temporal"] = n3_res

# ════════════════════════════════════════════════════════════════════════════
# Novelty 4: Self-Supervised Pre-training + Fine-tuning
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("NOVELTY 4: Self-Supervised Contrastive Pre-training")
print("="*60)
from ssl_pretrain import GraphContrastivePretrainer

# Use 40% of base_ds as "unlabeled" for pre-training, rest for fine-tuning
# (simulates scarce labeled data scenario)
np.random.seed(42)
idx_all     = np.random.permutation(len(base_ds))
n_unlabeled = int(0.4 * len(base_ds))
unlabeled   = [base_ds[i] for i in idx_all[:n_unlabeled]]
labeled     = [base_ds[i] for i in idx_all[n_unlabeled:]]

print(f"  Unlabeled (pre-train): {len(unlabeled)}")
print(f"  Labeled   (fine-tune): {len(labeled)}")

encoder     = TrafficGNN(input_dim=1500, hidden_dim=128, num_classes=4)
pretrainer  = GraphContrastivePretrainer(encoder, proj_dim=128, device=DEVICE)

pretrainer.pretrain(unlabeled, epochs=50, batch_size=64,
                    save_path="data/pretrained_encoder.pt")

# Fine-tune on labeled subset with 5-fold CV
def n4_model():
    m = TrafficGNN(input_dim=1500, hidden_dim=128, num_classes=4)
    if os.path.exists("data/pretrained_encoder.pt"):
        m.load_state_dict(
            torch.load("data/pretrained_encoder.pt", weights_only=True)
        )
    return m

n4_res = kfold_eval(labeled, n4_model, label="Novelty4-SSL")
all_results["novelty4_ssl"] = n4_res

# ════════════════════════════════════════════════════════════════════════════
# Novelty 5: Intrinsic Explainability
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("NOVELTY 5: Intrinsic Attention-Based Explainability")
print("="*60)
from explainable_gnn import ExplainableTrafficGNN, train_with_explanation_loss

def n5_model():
    return ExplainableTrafficGNN(input_dim=1500, hidden_dim=128,
                                  num_classes=4, edge_attr_dim=1,
                                  sparsity_lambda=1e-3)

# Custom training loop needed (uses explanation_loss)
def kfold_eval_explainable(dataset):
    labels  = np.array([int(g.y.item()) for g in dataset])
    ds_arr  = np.array(dataset, dtype=object)
    skf     = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    results = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(ds_arr, labels)):
        print(f"  Novelty5-Explainable | fold {fold+1}/{FOLDS} ...", end=" ", flush=True)
        model = n5_model().to(DEVICE)
        tr    = [dataset[i] for i in tr_idx]
        te    = [dataset[i] for i in te_idx]

        samp      = make_sampler([int(dataset[i].y.item()) for i in tr_idx])
        t_loader  = DataLoader(tr, batch_size=BATCH, sampler=samp)
        v_loader  = DataLoader(te, batch_size=BATCH, shuffle=False)
        opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

        best_f1, best_state = 0.0, None
        for ep in range(1, EPOCHS + 1):
            loss = train_with_explanation_loss(model, t_loader, opt, DEVICE)
            if ep % 20 == 0 or ep == EPOCHS:
                model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for batch in v_loader:
                        batch = batch.to(DEVICE)
                        preds.extend(model(batch).argmax(1).cpu().tolist())
                        trues.extend(batch.y.cpu().tolist())
                f1 = f1_score(trues, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1    = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in v_loader:
                batch = batch.to(DEVICE)
                preds.extend(model(batch).argmax(1).cpu().tolist())
                trues.extend(batch.y.cpu().tolist())

        res = {
            "acc":  accuracy_score(trues, preds),
            "prec": precision_score(trues, preds, average="macro", zero_division=0),
            "rec":  recall_score(trues, preds, average="macro", zero_division=0),
            "f1":   f1_score(trues, preds, average="macro", zero_division=0),
        }
        print(f"acc={res['acc']:.4f}  f1={res['f1']:.4f}")
        results.append(res)

        # Save explanation for the last fold
        if fold == FOLDS - 1:
            sample = te[0]
            sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long)
            expl = model.explain(sample.to(DEVICE))
            torch.save(expl, "data/sample_explanation.pt")
            print(f"  Sample explanation saved -> data/sample_explanation.pt")

    summary = {m: (np.mean([r[m] for r in results]),
                   np.std( [r[m] for r in results]))
               for m in ["acc","prec","rec","f1"]}
    print(f"  Novelty5 SUMMARY:")
    for m,(mu,sd) in summary.items():
        print(f"    {m.upper():4s}: {mu:.4f} +/- {sd:.4f}")
    return summary

n5_res = kfold_eval_explainable(base_ds)
all_results["novelty5_explainable"] = n5_res

# ════════════════════════════════════════════════════════════════════════════
# Novelty 6: Online Incremental Inference (Early Exit)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("NOVELTY 6: Online Incremental Inference (Early Exit)")
print("="*60)
from online_inference import benchmark_early_exit

# Load the best base model checkpoint from train.py (fold 1 by default)
base_model = TrafficGNN(input_dim=1500, hidden_dim=128, num_classes=4).to(DEVICE)
ckpt_path  = "data/best_model_fold1.pt"
if os.path.exists(ckpt_path):
    base_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"  Loaded checkpoint: {ckpt_path}")
else:
    print(f"  WARNING: {ckpt_path} not found -- run train.py first")
    print("  Using randomly initialized model (numbers will be low)")

# Benchmark across multiple confidence thresholds
n6_results = {}
test_subset = base_ds[: min(500, len(base_ds))]   # limit for speed

for thresh in [0.70, 0.80, 0.90, 0.95]:
    acc, avg_pkts, lat_red = benchmark_early_exit(
        base_model, test_subset, DEVICE, confidence_threshold=thresh
    )
    n6_results[thresh] = {"acc": acc, "avg_pkts": avg_pkts, "latency_reduction": lat_red}

all_results["novelty6_early_exit"] = n6_results

# ════════════════════════════════════════════════════════════════════════════
# Save all results for visualize_results.py
# ════════════════════════════════════════════════════════════════════════════
torch.save(all_results, "data/novelty_results.pt")
print("\n" + "="*60)
print("All novelty results saved -> data/novelty_results.pt")
print("Run: python visualize_results.py")
print("="*60)

# Print final summary table
print("\n FINAL COMPARISON TABLE (macro F1)")
print(f"  {'Method':<32}  {'F1 mean':>8}  {'F1 std':>8}")
print("  " + "-"*52)
mapping = {
    "novelty1_adaptive":    "Novelty 1: Adaptive threshold",
    "novelty2_hierarchical":"Novelty 2: Hierarchical graph",
    "novelty3_temporal":    "Novelty 3: Temporal edges",
    "novelty4_ssl":         "Novelty 4: SSL pre-training",
    "novelty5_explainable": "Novelty 5: Intrinsic explainability",
}
for key, name in mapping.items():
    res = all_results.get(key)
    if res is None:
        print(f"  {name:<32}  {'skipped':>8}")
    else:
        mu, sd = res["f1"]
        print(f"  {name:<32}  {mu:>8.4f}  {sd:>8.4f}")