"""Reproduce Table 4: Masking Evaluation.

For both full (hybrid) and bilinear checkpoints:
1. Load trained checkpoint
2. For each single modality (seq, text, struct, ppi): evaluate with other modalities masked
3. Output summary matching Table 4
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from mmfp.dataset import MultiModalDataset, collate_fn
from mmfp.evaluation import evaluate_with_cafa
from mmfp.models import create_model
from train import evaluate

ASPECTS = ["BPO", "CCO", "MFO"]
DATA_DIR = PROJECT_DIR / "data"
OBO_FILE = DATA_DIR / "go.obo"
IA_DIR = DATA_DIR
TEXT_EMBED_DIR = DATA_DIR / "embedding_cache" / "exp_text_embeddings_temporal"

MODALITY_CONDITIONS = [
    {"name": "seq_only", "kept_modality": "prott5", "mask": {"text": True, "struct": True, "ppi": True}},
    {"name": "text_only", "kept_modality": "text", "mask": {"prott5": True, "struct": True, "ppi": True}},
    {"name": "struct_only", "kept_modality": "struct", "mask": {"prott5": True, "text": True, "ppi": True}},
    {"name": "ppi_only", "kept_modality": "ppi", "mask": {"prott5": True, "text": True, "struct": True}},
]

CHECKPOINT_CONFIGS = [
    {
        "label": "hybrid",
        "checkpoint_root": PROJECT_DIR / "results" / "full_model" / "fusion_comparison" / "prott5",
        "fusion_type": "gated_bilinear",
        "use_late_fusion": True,
        "late_output_mode": "hybrid",
        "output_root": PROJECT_DIR / "results" / "masking_eval" / "hybrid",
    },
    {
        "label": "bilinear",
        "checkpoint_root": PROJECT_DIR / "results" / "full_model" / "fusion_comparison" / "prott5",
        "fusion_type": "gated_bilinear",
        "use_late_fusion": False,
        "late_output_mode": "hybrid",
        "output_root": PROJECT_DIR / "results" / "masking_eval" / "bilinear",
    },
]


def canonical_embedding_dirs() -> Dict[str, str]:
    return {
        "text": str(TEXT_EMBED_DIR),
        "prott5": str(DATA_DIR / "embedding_cache" / "prott5"),
        "esm": str(DATA_DIR / "embedding_cache" / "esm"),
        "struct": str(DATA_DIR / "embedding_cache" / "IF1"),
        "ppi": str(DATA_DIR / "embedding_cache" / "ppi"),
    }


def make_masked_dirs(condition: Dict, output_root: Path) -> Dict[str, str]:
    """Create embedding dirs with empty dirs for masked modalities."""
    full = canonical_embedding_dirs()
    result = {}
    for modality, path in full.items():
        if condition["mask"].get(modality, False):
            empty_dir = output_root / condition["name"] / f"_empty_{modality}"
            empty_dir.mkdir(parents=True, exist_ok=True)
            result[modality] = str(empty_dir)
        else:
            result[modality] = path
    return result


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_masked(
    config: Dict, condition: Dict, aspect: str
) -> Dict:
    """Evaluate a checkpoint with a specific modality masking condition."""
    output_dir = (
        config["output_root"] / condition["name"]
        / "fusion_comparison" / "prott5" / aspect / config["fusion_type"]
    )
    results_file = output_dir / "results.json"

    if results_file.exists():
        return load_json(results_file)

    checkpoint_path = (
        config["checkpoint_root"] / aspect / config["fusion_type"] / "best_model.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dirs = make_masked_dirs(condition, config["output_root"])

    train_dataset = MultiModalDataset(
        str(DATA_DIR), canonical_embedding_dirs(), "prott5", aspect, "train", normalize="standard"
    )
    test_dataset = MultiModalDataset(
        str(DATA_DIR),
        embedding_dirs,
        "prott5",
        aspect,
        "test",
        normalize="standard",
        norm_stats=train_dataset.norm_stats,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    model = create_model(
        fusion_type=config["fusion_type"],
        seq_dim=1024,
        text_dim=768,
        struct_dim=512,
        ppi_dim=512,
        hidden_dim=512,
        num_go_terms=train_dataset.labels.shape[1],
        dropout=0.4,
        modality_dropout=0.1,
        use_late_fusion=config["use_late_fusion"],
        late_output_mode=config["late_output_mode"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss().to(device)
    test_metrics = evaluate(model, test_loader, criterion, device, compute_weight_stats_detailed=True)
    go_terms = load_json(DATA_DIR / f"{aspect}_go_terms.json")
    ia_file = IA_DIR / f"{aspect}_ia.txt"
    cafa_metrics = evaluate_with_cafa(
        model=model,
        loader=test_loader,
        device=device,
        protein_ids=test_dataset.protein_ids.tolist(),
        go_terms=go_terms,
        obo_file=str(OBO_FILE),
        output_dir=str(output_dir / "cafa_eval"),
        model_name=f"prott5_{condition['name']}_{config['label']}_temporal",
        train_labels=train_dataset.labels,
        ia_file=str(ia_file) if ia_file.exists() else None,
    )

    n_total = len(test_dataset.protein_ids)
    n_eval = sum(1 for pid in test_dataset.protein_ids if any(
        (Path(embedding_dirs[condition["kept_modality"]]) / f"{pid}.npy").exists()
        for _ in [None]
    ))

    result = {
        "aspect": aspect,
        "condition": condition["name"],
        "protocol": "paper_conditional_single_modality",
        "test_fmax": float(test_metrics["fmax"]),
        "cafa_fmax": float(cafa_metrics["fmax"]),
        "cafa_wfmax": float(cafa_metrics["wfmax"]),
        "cafa_smin": float(cafa_metrics["smin"]),
        "n_eval": n_eval,
        "n_total": n_total,
        "coverage": n_eval / n_total if n_total > 0 else 0,
        "source_checkpoint": str(checkpoint_path),
        "kept_modality": condition["kept_modality"],
        "weight_seq": float(test_metrics.get("weight_seq", 0)),
        "weight_text": float(test_metrics.get("weight_text", 0)),
        "weight_struct": float(test_metrics.get("weight_struct", 0)),
        "weight_ppi": float(test_metrics.get("weight_ppi", 0)),
    }
    if "late_fusion_lambda" in test_metrics:
        result["late_fusion_lambda"] = float(test_metrics["late_fusion_lambda"])

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_file, result)
    return result


def main() -> int:
    for config in CHECKPOINT_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Masking evaluation: {config['label']}")
        print(f"{'='*60}")

        summary_rows: List[Dict] = []
        for condition in MODALITY_CONDITIONS:
            for aspect in ASPECTS:
                print(f"  {condition['name']} / {aspect}...")
                result = evaluate_masked(config, condition, aspect)
                summary_rows.append(result)

        csv_path = config["output_root"] / "single_modality_summary.csv"
        json_path = config["output_root"] / "single_modality_summary.json"
        save_csv(csv_path, summary_rows)
        save_json(json_path, {"rows": summary_rows})
        print(f"  Summary: {csv_path}")

    print("\nMasking evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
