"""Run the full-modal temporal ablation suite."""

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
from train import evaluate, train_fusion_model


ASPECTS = ["BPO", "CCO", "MFO"]
DATA_DIR = PROJECT_DIR / "data"
OBO_FILE = DATA_DIR / "go.obo"
IA_DIR = DATA_DIR
TEXT_EMBED_DIR = DATA_DIR / "embedding_cache" / "exp_text_embeddings_temporal"
OUTPUT_ROOT = PROJECT_DIR / "results" / "ablation"
SUMMARY_CSV = OUTPUT_ROOT / "ablation_summary.csv"
SUMMARY_JSON = OUTPUT_ROOT / "ablation_summary.json"
SUMMARY_MD = OUTPUT_ROOT / "ablation_summary.md"

FULL_CONTROL_ROOT = (
    PROJECT_DIR
    / "results"
    / "full_model_eval"
    / "eval_only"
    / "fusion_comparison"
    / "prott5"
)
EARLY_CONTROL_ROOT = (
    PROJECT_DIR
    / "results"
    / "masking_eval"
    / "bilinear_subset"
    / "full"
    / "fusion_comparison"
    / "prott5"
)

EXPECTED_FULL = {
    "BPO": {"cafa_fmax": 0.601, "cafa_wfmax": 0.515},
    "CCO": {"cafa_fmax": 0.706, "cafa_wfmax": 0.566},
    "MFO": {"cafa_fmax": 0.702, "cafa_wfmax": 0.605},
}
EXPECTED_EARLY = {
    "BPO": {"cafa_fmax": 0.593, "cafa_wfmax": 0.504},
    "CCO": {"cafa_fmax": 0.703, "cafa_wfmax": 0.560},
    "MFO": {"cafa_fmax": 0.691, "cafa_wfmax": 0.599},
}

ABLATIONS = [
    {
        "ablation_name": "full",
        "config_label": "gated_bilinear_hybrid",
        "fusion_type": "gated_bilinear",
        "use_late_fusion": True,
        "late_output_mode": "hybrid",
        "source": "reuse_full_control",
    },
    {
        "ablation_name": "full_wo_aux_late",
        "config_label": "gated_bilinear_early",
        "fusion_type": "gated_bilinear",
        "use_late_fusion": False,
        "late_output_mode": "hybrid",
        "source": "reuse_early_control",
    },
    {
        "ablation_name": "full_wo_early_fusion",
        "config_label": "aux_only_uniform",
        "fusion_type": "aux_only",
        "use_late_fusion": True,
        "late_output_mode": "aux_only",
        "source": "train",
    },
    {
        "ablation_name": "concat_only",
        "config_label": "concat_early",
        "fusion_type": "concat",
        "use_late_fusion": False,
        "late_output_mode": "hybrid",
        "source": "train",
    },
    {
        "ablation_name": "concat_plus_aux_late",
        "config_label": "concat_hybrid",
        "fusion_type": "concat",
        "use_late_fusion": True,
        "late_output_mode": "hybrid",
        "source": "train",
    },
]


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Aspect | Ablation | Config | Test Fmax | CAFA Fmax | CAFA WFmax | Delta Fmax | Delta WFmax |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {aspect} | {ablation_name} | {config_label} | {test_fmax:.4f} | "
            "{cafa_fmax:.3f} | {cafa_wfmax:.3f} | {delta_vs_full_cafa_fmax:+.3f} | "
            "{delta_vs_full_cafa_wfmax:+.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def canonical_embedding_dirs() -> Dict[str, str]:
    return {
        "text": str(TEXT_EMBED_DIR),
        "prott5": str(DATA_DIR / "embedding_cache" / "prott5"),
        "esm": str(DATA_DIR / "embedding_cache" / "esm"),
        "struct": str(DATA_DIR / "embedding_cache" / "IF1"),
        "ppi": str(DATA_DIR / "embedding_cache" / "ppi"),
    }


def create_eval_model(spec: Dict, num_go_terms: int):
    return create_model(
        fusion_type=spec["fusion_type"],
        seq_dim=1024,
        text_dim=768,
        struct_dim=512,
        ppi_dim=512,
        hidden_dim=512,
        num_go_terms=num_go_terms,
        dropout=0.4,
        modality_dropout=0.1,
        use_late_fusion=spec["use_late_fusion"],
        late_output_mode=spec["late_output_mode"],
    )


def verify_control_metrics(root: Path, expected: Dict[str, Dict[str, float]]) -> None:
    for aspect, metrics in expected.items():
        result_path = root / aspect / "gated_bilinear" / "results.json"
        result = load_json(result_path)
        for key, expected_value in metrics.items():
            actual = float(result[key])
            if abs(actual - expected_value) > 5e-4:
                raise ValueError(
                    f"Control mismatch for {aspect} at {result_path}: {key}={actual} expected {expected_value}"
                )


def evaluate_existing_ablation(spec: Dict, aspect: str) -> Dict:
    output_dir = OUTPUT_ROOT / spec["ablation_name"] / "fusion_comparison" / "prott5" / aspect / spec["fusion_type"]
    results_file = output_dir / "results.json"
    checkpoint_path = output_dir / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {spec['ablation_name']} {aspect}: {checkpoint_path}")

    result = load_json(results_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dirs = canonical_embedding_dirs()

    train_dataset = MultiModalDataset(
        str(DATA_DIR), embedding_dirs, "prott5", aspect, "train", normalize="standard"
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

    model = create_eval_model(spec, num_go_terms=train_dataset.labels.shape[1]).to(device)
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
        model_name=f"prott5_{spec['ablation_name']}_temporal",
        train_labels=train_dataset.labels,
        ia_file=str(ia_file) if ia_file.exists() else None,
    )

    result.update(
        {
            "ablation_name": spec["ablation_name"],
            "config_label": spec["config_label"],
            "fusion_type": spec["fusion_type"],
            "use_late_fusion": spec["use_late_fusion"],
            "late_output_mode": spec["late_output_mode"],
            "test_fmax": float(test_metrics["fmax"]),
            "test_micro_auprc": float(test_metrics["micro_auprc"]),
            "test_macro_auprc": float(test_metrics["macro_auprc"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall": float(test_metrics["recall"]),
            "weight_seq": float(test_metrics["weight_seq"]),
            "weight_text": float(test_metrics["weight_text"]),
            "weight_struct": float(test_metrics["weight_struct"]),
            "weight_ppi": float(test_metrics["weight_ppi"]),
            "cafa_fmax": float(cafa_metrics["fmax"]),
            "cafa_wfmax": float(cafa_metrics["wfmax"]),
            "cafa_smin": float(cafa_metrics["smin"]),
            "cafa_precision": float(cafa_metrics["precision"]),
            "cafa_recall": float(cafa_metrics["recall"]),
        }
    )
    if "late_fusion_lambda" in test_metrics:
        result["late_fusion_lambda"] = float(test_metrics["late_fusion_lambda"])

    save_json(results_file, result)
    return result


def run_train_ablation(spec: Dict, aspect: str) -> Dict:
    output_base = OUTPUT_ROOT / spec["ablation_name"]
    results_file = output_base / "fusion_comparison" / "prott5" / aspect / spec["fusion_type"] / "results.json"
    if results_file.exists():
        existing = load_json(results_file)
        if existing.get("cafa_fmax") is not None and existing.get("cafa_wfmax") is not None:
            return existing
        return evaluate_existing_ablation(spec, aspect)

    result = train_fusion_model(
        seq_model="prott5",
        aspect=aspect,
        fusion_type=spec["fusion_type"],
        data_dir=str(DATA_DIR),
        embedding_dirs=canonical_embedding_dirs(),
        obo_file=str(OBO_FILE),
        modality_dropout=0.1,
        output_base=str(output_base),
        use_late_fusion=spec["use_late_fusion"],
        late_output_mode=spec["late_output_mode"],
        aux_loss_weight=0.8,
        num_workers=0,
        ia_file=str(IA_DIR / f"{aspect}_ia.txt") if (IA_DIR / f"{aspect}_ia.txt").exists() else None,
        seed=42,
        ablation_name=spec["ablation_name"],
        config_label=spec["config_label"],
    )
    return result


def load_reused_result(spec: Dict, aspect: str) -> Dict:
    if spec["source"] == "reuse_full_control":
        path = FULL_CONTROL_ROOT / aspect / "gated_bilinear" / "results.json"
    elif spec["source"] == "reuse_early_control":
        path = EARLY_CONTROL_ROOT / aspect / "gated_bilinear" / "results.json"
    else:
        raise ValueError(f"Unsupported source {spec['source']}")
    result = load_json(path)
    result = dict(result)
    result["ablation_name"] = spec["ablation_name"]
    result["config_label"] = spec["config_label"]
    result["fusion_type"] = spec["fusion_type"]
    result["use_late_fusion"] = spec["use_late_fusion"]
    result["late_output_mode"] = spec["late_output_mode"]
    result["results_json"] = str(path)
    return result


def main() -> int:
    verify_control_metrics(FULL_CONTROL_ROOT, EXPECTED_FULL)
    verify_control_metrics(EARLY_CONTROL_ROOT, EXPECTED_EARLY)

    by_aspect: Dict[str, List[Dict]] = {aspect: [] for aspect in ASPECTS}

    for spec in ABLATIONS:
        for aspect in ASPECTS:
            if spec["source"] == "train":
                result = run_train_ablation(spec, aspect)
                result["results_json"] = str(
                    OUTPUT_ROOT
                    / spec["ablation_name"]
                    / "fusion_comparison"
                    / "prott5"
                    / aspect
                    / spec["fusion_type"]
                    / "results.json"
                )
            else:
                result = load_reused_result(spec, aspect)
            by_aspect[aspect].append(result)

    summary_rows: List[Dict] = []
    for aspect in ASPECTS:
        full_row = next(row for row in by_aspect[aspect] if row["ablation_name"] == "full")
        full_fmax = float(full_row["cafa_fmax"])
        full_wfmax = float(full_row["cafa_wfmax"])
        for row in by_aspect[aspect]:
            summary_rows.append(
                {
                    "aspect": aspect,
                    "ablation_name": row["ablation_name"],
                    "config_label": row["config_label"],
                    "fusion_type": row["fusion_type"],
                    "use_late_fusion": bool(row["use_late_fusion"]),
                    "late_output_mode": row["late_output_mode"],
                    "test_fmax": float(row["test_fmax"]),
                    "cafa_fmax": float(row["cafa_fmax"]),
                    "cafa_wfmax": float(row["cafa_wfmax"]),
                    "delta_vs_full_cafa_fmax": float(row["cafa_fmax"]) - full_fmax,
                    "delta_vs_full_cafa_wfmax": float(row["cafa_wfmax"]) - full_wfmax,
                    "results_json": row["results_json"],
                }
            )

    save_csv(SUMMARY_CSV, summary_rows)
    save_markdown(SUMMARY_MD, summary_rows)
    save_json(
        SUMMARY_JSON,
        {
            "protocol": {
                "seq_model": "prott5",
                "seed": 42,
                "text_embedding_dir": str(TEXT_EMBED_DIR),
                "full_control_root": str(FULL_CONTROL_ROOT),
                "early_control_root": str(EARLY_CONTROL_ROOT),
            },
            "rows": summary_rows,
        },
    )

    print(f"Ablation suite complete. Summary: {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
