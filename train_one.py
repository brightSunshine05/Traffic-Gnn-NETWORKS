import sys, os, time, warnings, torch, numpy as np
warnings.filterwarnings("ignore")

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

from model import TrafficGNN
from adaptive_graph_builder import AdaptiveTrafficGNN
from hierarchical_graph import HierarchicalTrafficGNN
from temporal_edges import TemporalTrafficGNN
from explainable_gnn import ExplainableTrafficGNN, train_with_explanation_loss

FOLDS = 5
EPOCHS = 35
HIDDEN_DIM = 128
BATCH = 64
LR = 1e-3
WD = 1e-5
DEVICE = torch.device("cpu")

which = sys.argv[1]

DATASETS = {
    "base": ("data/nonvpn_graph_dataset.pt", lambda: TrafficGNN(1500, HIDDEN_DIM, 4), False),
    "n1":   ("data/nonvpn_adaptive_dataset.pt", lambda: AdaptiveTrafficGNN(1500, HIDDEN_DIM, 4), False),
    "n2":   ("data/nonvpn_hierarchical_dataset.pt", lambda: HierarchicalTrafficGNN(1500, HIDDEN_DIM, 4), False),
    "n3":   ("data/nonvpn_temporal_dataset.pt", lambda: TemporalTrafficGNN(1500, 64, 4, 4), False),
    "n5":   ("data/nonvpn_graph_dataset.pt", lambda: ExplainableTrafficGNN(1500, HIDDEN_DIM, 4, 1, 1e-3), True),
}
path, model_fn, use_expl = DATASETS[which]
dataset = torch.load(path, weights_only=False)
print(f"[{which}] loaded {len(dataset)} graphs from {path}")


def make_sampler(labels):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = {c: total / counts[c] for c in counts}
    sw = [weights[l] for l in labels]
    return WeightedRandomSampler(sw, len(sw), replacement=True)


def run_fold(model, train_data, test_data):
    labels = [int(g.y.item()) for g in train_data]
    sampler = make_sampler(labels)
    t_loader = DataLoader(train_data, batch_size=BATCH, sampler=sampler)
    v_loader = DataLoader(test_data, batch_size=BATCH, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    crit = torch.nn.CrossEntropyLoss()

    best_f1, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        if use_expl:
            train_with_explanation_loss(model, t_loader, opt, DEVICE)
        else:
            model.train()
            for batch in t_loader:
                opt.zero_grad()
                out = model(batch)
                loss = crit(out, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
        if ep % 5 == 0 or ep == EPOCHS:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in v_loader:
                    preds.extend(model(batch).argmax(1).tolist())
                    trues.extend(batch.y.tolist())
            f1 = f1_score(trues, preds, average="macro", zero_division=0)
            if f1 >= best_f1:
                best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in v_loader:
            preds.extend(model(batch).argmax(1).tolist())
            trues.extend(batch.y.tolist())
    return {
        "acc": accuracy_score(trues, preds),
        "prec": precision_score(trues, preds, average="macro", zero_division=0),
        "rec": recall_score(trues, preds, average="macro", zero_division=0),
        "f1": f1_score(trues, preds, average="macro", zero_division=0),
    }, trues, preds


t0 = time.time()
labels = np.array([int(g.y.item()) for g in dataset])
ds_arr = np.array(dataset, dtype=object)
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
results, last_trues, last_preds = [], None, None

for fold, (tr_idx, te_idx) in enumerate(skf.split(ds_arr, labels)):
    model = model_fn().to(DEVICE)
    res, trues, preds = run_fold(model, [dataset[i] for i in tr_idx], [dataset[i] for i in te_idx])
    print(f"  fold {fold + 1}/{FOLDS}: acc={res['acc']:.4f} f1={res['f1']:.4f}  ({time.time() - t0:.1f}s elapsed)")
    results.append(res)
    last_trues, last_preds = trues, preds

summary = {m: (float(np.mean([r[m] for r in results])), float(np.std([r[m] for r in results])))
           for m in ["acc", "prec", "rec", "f1"]}
print(f"[{which}] SUMMARY:", "  ".join(f"{m.upper()}={mu:.4f}+/-{sd:.4f}" for m, (mu, sd) in summary.items()))
print(f"[{which}] total time: {time.time() - t0:.1f}s")

results_path = "data/novelty_results.pt"
all_results = torch.load(results_path, weights_only=False) if os.path.exists(results_path) else {}
all_results[which] = summary
if which in ("base", "n5"):
    all_results[f"{which}_confusion"] = {"y_true": last_trues, "y_pred": last_preds}
torch.save(all_results, results_path)
print(f"Saved -> {results_path}")

from plot_results import plot_all
plot_all(all_results)
