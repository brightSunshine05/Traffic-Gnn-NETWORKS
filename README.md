# Network Traffic Classification — Final Review Submission

Modified version of the senior's codebase (originally
`Network_Traffic_Classification_-_Rahul_25MAI0020.zip`). **See
`CHANGES.md` for a full, itemised list of every modification and why it
was made** — that file is the primary record of what changed in this
review.

## Layout
```
src/
  preprocessing.py              Real pcap parsing (scapy) -- UNCHANGED
  graph_builder.py               Base paper graphs (Algorithms 1-5) -- UNCHANGED
  model.py                       TrafficGNN, 5-layer GraphConv -- UNCHANGED
  adaptive_threshold.py          N1 building blocks -- UNCHANGED
  adaptive_graph_builder.py      N1 PROPERLY WIRED END-TO-END -- NEW (see CHANGES.md #1)
  hierarchical_graph.py          N2 hierarchical graph -- 1-line dtype fix (CHANGES.md #2)
  temporal_edges.py              N3 temporal edge attributes -- UNCHANGED
  explainable_gnn.py             N5 intrinsic attention -- UNCHANGED
  ssl_pretrain.py                N4 self-supervised pre-training -- 1-line dir-creation fix (CHANGES.md #3)
  online_inference.py            N6 early-exit streaming inference -- UNCHANGED
  synthetic_data.py              NEW -- class-conditioned synthetic traffic fallback (CHANGES.md #4)
  build_dataset.py               EDITED -- auto-fallback to synthetic data + builds N1's dataset
  build_hierarchical_dataset.py  EDITED -- same auto-fallback
  train.py                       Base model training (paper hyperparameters) -- UNCHANGED
  train_novalties.py             EDITED -- N1 section now uses the real wiring
  run_end_to_end_demo.py         NEW -- one-command demo of base + N1,N2,N3,N5,N6 (CHANGES.md #5)
  visualize_results.py           Plotting utilities -- UNCHANGED
outputs/
  final_run_log.txt              Console output backing every number in CHANGES.md #6
  final_confusion_matrix.png, final_model_comparison.png
  novelty_results.pt             Raw metrics (PyTorch pickle) for all 6 novelties
CHANGES.md                       Full changelog -- read this first
```

## How to run

**With real ISCX pcap captures** (once available): put them under
`src/data/nonvpn/<chat|email|streaming|voip>/*.pcap` and run, in order:
```
cd src
python3 build_dataset.py              # builds base, temporal, adaptive datasets
python3 build_hierarchical_dataset.py # builds the hierarchical (N2) dataset
python3 train.py                      # base model, paper hyperparameters (500 epochs, 5-fold CV)
python3 train_novalties.py            # all six novelties, paper hyperparameters
```
No code changes are needed to switch from synthetic to real data — see
CHANGES.md item 4.

**Without real captures** (this submission's validation run):
```
cd src
python3 run_end_to_end_demo.py   # also draws all graphs into plots/
```
Builds synthetic datasets, trains the base model and N1/N2/N3/N5 (reduced
epoch budget for CPU runtime), benchmarks N6, and writes plots to `plots/`.
This reproduces the numbers in `CHANGES.md` item 6 and `outputs/final_run_log.txt`.

## Dependencies
```
pip install torch torch_geometric scapy scikit-learn matplotlib tqdm
```


See HOW_TO_RUN.md for step-by-step instructions. Graphs are generated into src/plots/ when you run the code (see CHANGES.md item 7).
