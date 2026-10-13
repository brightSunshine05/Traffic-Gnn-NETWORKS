import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from collections import Counter
from model import TrafficGNN

print("Train script started...")

# -----------------------------------
# Paper section 4.2.2 hyperparameters
# -----------------------------------
BATCH_SIZE    = 256
EPOCHS        = 500
LR            = 0.001
BETAS         = (0.9, 0.999)
EPS           = 1e-8
WEIGHT_DECAY  = 1e-5
N_FOLDS       = 5       # paper: stratified 5-fold cross-validation
NUM_CLASSES   = 4
INPUT_DIM     = 1500
HIDDEN_DIM    = 128


# -----------------------------------
# Load dataset
# -----------------------------------
print("Loading dataset...")
dataset = torch.load("data/nonvpn_graph_dataset.pt", weights_only=False)
print(f"Total graphs: {len(dataset)}")

# Extract labels as a flat list for stratified splitting
labels = [int(g.y.item()) for g in dataset]
label_counts = Counter(labels)
print("Class distribution:", dict(sorted(label_counts.items())))


# -----------------------------------
# Build a WeightedRandomSampler
# to handle class imbalance
# (paper: weighted random sampler
# for flow classification)
# -----------------------------------
def make_weighted_sampler(subset_labels):
    counts = Counter(subset_labels)
    total = sum(counts.values())
    class_weights = {c: total / counts[c] for c in counts}
    sample_weights = [class_weights[l] for l in subset_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# -----------------------------------
# Single epoch training
# -----------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# -----------------------------------
# Evaluation -- returns all metrics
# -----------------------------------
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, prec, rec, f1, all_labels, all_preds


# -----------------------------------
# Stratified 5-fold cross-validation
# Paper section 4.2.2
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset_arr = np.array(dataset, dtype=object)
labels_arr  = np.array(labels)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(dataset_arr, labels_arr)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold+1}/{N_FOLDS}")
    print(f"{'='*50}")

    train_data = [dataset[i] for i in train_idx]
    test_data  = [dataset[i] for i in test_idx]

    train_labels = [labels[i] for i in train_idx]

    # Weighted sampler to handle class imbalance (paper section 4.2.2)
    sampler = make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        sampler=sampler,       # replaces shuffle=True
    )
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Fresh model per fold
    model = TrafficGNN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(device)

    # Paper optimizer: Adam with decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_f1   = 0.0
    best_acc  = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate every 10 epochs to save time; always evaluate last epoch
        if epoch % 10 == 0 or epoch == EPOCHS:
            acc, prec, rec, f1, _, _ = evaluate(model, test_loader, device)

            if f1 > best_f1:
                best_f1    = f1
                best_acc   = acc
                best_epoch = epoch
                # Save best checkpoint for this fold
                torch.save(model.state_dict(), f"data/best_model_fold{fold+1}.pt")

            print(f"  Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Acc: {acc:.4f} | Prec: {prec:.4f} | "
                  f"Rec: {rec:.4f} | F1: {f1:.4f}")

    # Load best checkpoint and get final report
    model.load_state_dict(torch.load(f"data/best_model_fold{fold+1}.pt", weights_only=True))
    acc, prec, rec, f1, y_true, y_pred = evaluate(model, test_loader, device)

    print(f"\nFold {fold+1} best @ epoch {best_epoch}: "
          f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    print(classification_report(y_true, y_pred,
                                 target_names=["chat", "email", "streaming", "voip"],
                                 digits=4))

    fold_results.append({"acc": acc, "prec": prec, "rec": rec, "f1": f1})


# -----------------------------------
# Aggregate cross-validation results
# -----------------------------------
print("\n" + "="*50)
print("CROSS-VALIDATION SUMMARY")
print("="*50)

for metric in ["acc", "prec", "rec", "f1"]:
    vals = [r[metric] for r in fold_results]
    print(f"  {metric.upper():5s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")