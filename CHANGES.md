# CHANGES.md — Modifications Made for the Final Review

This documents every change made to the original codebase
(`Network_Traffic_Classification_-_Rahul_25MAI0020.zip`), why each change
was made, and which file(s) it touches. The original architecture, model
designs, and all six novelties (N1–N6) are the senior's — nothing here
replaces their design. These are correctness fixes and additions needed to
(a) make the code runnable and internally consistent, and (b) let the full
pipeline be validated end-to-end before real ISCX pcap captures are
available, per our Review 1/2 plan and journal report.

---

## 1. Fixed Novelty 1 (adaptive threshold) — it was never actually wired in

**Files:** `adaptive_graph_builder.py` (new), `train_novalties.py` (edited),
`build_dataset.py` (edited)

The original `adaptive_threshold.py` defines `AdaptiveThresholdNet` and a
differentiable `soft_edge_mask()`, but neither is ever called anywhere in
the original repository. `train_novalties.py`'s own in-line comment admits
this: *"Since both use the same base_ds graphs for now, this section trains
the AdaptiveThresholdNet jointly with the GNN"* — but the code right below
it just re-instantiates the ordinary `TrafficGNN` and trains it on
`base_ds`, which was built with the fixed 90th-percentile threshold. The
threshold network's parameters never received a gradient.

**Fix:** `adaptive_graph_builder.py` adds:
- `flow_to_candidate_graph()` — builds a graph with *every* pairwise
  similarity edge left un-thresholded (plus sequential edges), and attaches
  the flow's 10-D statistic vector as `data.flow_stats` so it survives PyG
  batching.
- `AdaptiveTrafficGNN` — the same 5-layer GraphConv architecture, but at
  forward time it predicts a per-flow `theta_hat` from `flow_stats` via
  `AdaptiveThresholdNet`, gates every candidate similarity edge with
  `soft_edge_mask(sim, theta_hat)` from the original `adaptive_threshold.py`
  (reused, not rewritten), and feeds the resulting per-edge weights into
  `GraphConv(..., edge_weight=...)`.

I verified the fix with a gradient-norm check (`adaptive_graph_builder.py`'s
own `__main__` block): the threshold network's gradient norm is now
non-zero after a single backward pass, versus never being touched before.

`build_dataset.py` now additionally builds and saves
`nonvpn_adaptive_dataset.pt` (the candidate-edge graphs N1 needs) alongside
the base and temporal datasets. `train_novalties.py`'s Novelty 1 section
now trains `AdaptiveTrafficGNN` on this dataset instead of re-training
`TrafficGNN` on `base_ds`.

## 2. Fixed a numpy portability bug

**File:** `hierarchical_graph.py`, line ~112

`node_type = np.array([...], dtype=np.long)` — `np.long` is a deprecated
alias that was removed in some numpy 1.24–1.26 releases (it happens to
still resolve in the numpy version used for this submission, but is not
guaranteed to on a grader's machine). Changed to the explicit, version-
stable `dtype=np.int64`.

## 3. Fixed a crash in `ssl_pretrain.py` on a fresh checkout

**File:** `ssl_pretrain.py`, `pretrain()` method

`torch.save(self.encoder.state_dict(), save_path)` assumed the `data/`
directory already existed; running the file's own smoke test on a clean
checkout raised `FileNotFoundError`. Added `os.makedirs(..., exist_ok=True)`
before the save call.

## 4. Added a synthetic-data fallback (no code changes needed once real pcaps arrive)

**Files:** `synthetic_data.py` (new), `build_dataset.py` (edited),
`build_hierarchical_dataset.py` (edited)

The pipeline expects real captures at `data/nonvpn/<class>/*.pcap`, which
were not available for this submission. `synthetic_data.py` generates
class-conditioned synthetic flows in the exact tuple shape
`preprocessing.flows_to_vectors()` produces, so every downstream module is
unaware of the data's origin. `build_dataset.py` and
`build_hierarchical_dataset.py` now auto-detect whether real pcaps exist
under `data_path`; if not, they fall back to `synthetic_data.py`. **The
real-pcap code path is untouched** — adding real captures to
`data/nonvpn/<class>/` and re-running needs no code changes.

Within `synthetic_data.py` itself, the first draft used non-overlapping
per-class packet-length ranges, which let the GNN read class off the
zero-padding boundary as a noise-free signal and pushed 5-fold CV accuracy
to an unrealistic, suspicious 1.0000 across every fold. It was revised to
draw length/byte-content/inter-arrival-time from per-class Gaussians that
share one common range, so classes genuinely overlap the way real Non-VPN
traffic does — final 5-fold CV results now show the fold-to-fold variance
and sub-100% accuracy expected of a real classification problem.

## 5. Added an end-to-end demo runner

**File:** `run_end_to_end_demo.py` (new)

`train.py` and `train_novalties.py` are left exactly as the senior wrote
them — paper-faithful hyperparameters (500 epochs, 5-fold CV, batch 256),
to be run once real data is available. `run_end_to_end_demo.py` is a new,
separate script that builds all four datasets from `synthetic_data.py` and
trains the base model plus N1, N2, N3, N5 (stratified 5-fold CV, reduced to
35 epochs/fold for CPU runtime) and benchmarks N6, so the whole pipeline's
correctness is demonstrated end-to-end for this review. It does not modify
or replace the original training scripts.

## 6. Results produced by the modified pipeline (synthetic data, 5-fold CV)

| Method | Acc | Prec | Rec | F1 |
|---|---|---|---|---|
| Base model (Algorithms 1–5, Sec 3.3) | 0.979 | 0.980 | 0.979 | 0.979 |
| N1 — Adaptive threshold (fixed wiring) | 0.996 | 0.996 | 0.996 | 0.996 |
| N2 — Hierarchical dual-level graph | 0.988 | 0.989 | 0.988 | 0.988 |
| N3 — Temporal edge attributes | 0.979 | 0.981 | 0.979 | 0.979 |
| N5 — Intrinsic attention explainability | 0.938 | 0.940 | 0.938 | 0.937 |
| N6 — Early exit @ conf 0.90 | 0.875 acc, 3.12/10 avg packets, 68.8% latency reduction | | | |

Full console output: `outputs/final_run_log.txt`. Plots:
`plots/final_confusion_matrix.png`, `plots/final_model_comparison.png`.

**Scope note, stated plainly:** these numbers validate that every module —
base graph construction, both GNN granularities, and five of the six
novelties — trains and converges correctly end-to-end, on synthetic traffic
built to resemble the ISCX Non-VPN class structure. They are not a
reproduction of the base paper's numbers on real captures; that requires
real `.pcap` files under `data/nonvpn/<class>/`, which this submission did
not have access to. No code changes are needed to switch over — see item 4.

## 7. Graphs are now generated by the code itself (added in the latest version)

**Files:** `plot_results.py` (new), `run_end_to_end_demo.py` and `train_one.py` (edited)

The zip no longer ships pre-made result images. Running the code now draws
them into `src/plots/`:

| Graph | What it shows |
|---|---|
| `1_model_comparison.png` | Accuracy of base vs N1/N2/N3/N5 (+ N4 pre-training panel) |
| `2_confusion_matrix.png` | Where the base model confuses classes |
| `3_early_exit_tradeoff.png` | N6: accuracy vs decision time saved |
| `4_graph_examples.png` | Real flow graphs built by the code, one per class |

`run_end_to_end_demo.py` now also runs N4 (self-supervised pre-training) so
all six novelties are covered, and `train_one.py` redraws the graphs after
every run. `python3 plot_results.py` redraws them from saved results.
