#!/usr/bin/env bash
"""
train_script.py

Launches a grid of training runs over multiple aspects, folds, learning rates, and embedding types,
and records results to a summary CSV.

Usage:
    python train_script.py

Dependencies:
    - Python standard library: subprocess, itertools, time, csv, os, sys
    - Assumes train.py is in the same directory and configured via Hydra.
"""
    
import subprocess
import itertools
import time
import csv
import os
import sys

# Experiment configuration
aspects = ["mfo" , "cco", "bpo"]
folds   = range(5)
lrs     = [0.01, 0.005, 0.001]
embs    = ["esm_mean",  "esm", "text"]
# embs    = ["mmsite"]
# "mmsite", 
# Per‑ontology label count  
dim_by_aspect = {"bpo": 1302, "cco": 453, "mfo": 483}

# Summary CSV path
csv_path = "experiments_summary.csv"

# Initialize CSV file with header if not exists
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "aspect", "fold", "lr", "embedding",
            "out_dir", "return_code", "duration_s"
        ])

# Iterate through the experiment grid
for aspect, fold, lr, emb in itertools.product(aspects, folds, lrs, embs):
    out_dir = os.path.join("runs", aspect, emb, f"lr{lr}", f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "train.py",
        "--config-name", aspect,
        f"dataset.embedding_type={emb}",
        f"fold={fold}",
        f"optim.lr={lr}",
        f"log.out_dir={out_dir}",   
        f"model.output_dim={dim_by_aspect[aspect]}"
    ]

    print(f">>> Running: aspect={aspect}, fold={fold}, emb={emb}, lr={lr}")
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    print(f"    Return code: {result.returncode}, Duration: {elapsed:.1f}s")

    # Append results to CSV
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            aspect, fold, lr, emb,
            out_dir, result.returncode, f"{elapsed:.1f}"
        ])

    # Stop on error
    if result.returncode != 0:
        print("ERROR: Non-zero return code, aborting further runs.")
        sys.exit(result.returncode)

print("All experiments completed. Summary in", csv_path)
# Pv;5QUGE>ud6