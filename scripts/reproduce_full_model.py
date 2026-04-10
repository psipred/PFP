"""Canonical reproduction runner for the full-model temporal recipe."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from mmfp.dataset import MultiModalDataset, collate_fn
from mmfp.evaluation import evaluate_with_cafa
from mmfp.models import create_model
from train import evaluate
from extract_uniprot_text import (
    CURRENT_OUTPUT_FILE,
    DATA_DIR,
    HISTORICAL_OUTPUT_FILE,
    TEMPORAL_DIR,
    TEMPORAL_PUNCT_FILE,
    MIXED_OUTPUT_FILE,
    TEXT_BUNDLE_METADATA,
    build_historical_punct_v1_test_tsv,
    build_mixed_temporal_tsv,
    clean_historical_description_artifacts,
    get_split_protein_ids,
    load_protein_descriptions,
    materialize_current_text_source,
)

OBO_FILE = DATA_DIR / "go.obo"
CURRENT_EMBED_DIR = DATA_DIR / "embedding_cache" / "exp_text_embeddings"
MIXED_EMBED_DIR = DATA_DIR / "embedding_cache" / "exp_text_embeddings_temporal"
OUTPUT_ROOT = PROJECT_DIR / "results" / "full_model_eval"
SUMMARY_CSV = OUTPUT_ROOT / "reproduction_summary.csv"
SUMMARY_JSON = OUTPUT_ROOT / "reproduction_summary.json"
EVAL_OUTPUT_ROOT = OUTPUT_ROOT / "eval_only"
EXPECTED = {
    "BPO": {"fmax": 0.601, "wfmax": 0.515},
    "CCO": {"fmax": 0.706, "wfmax": 0.566},
    "MFO": {"fmax": 0.702, "wfmax": 0.605},
}
ASPECTS = ["BPO", "CCO", "MFO"]
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_LENGTH = 512
CURRENT_RESULTS = {
    aspect: PROJECT_DIR
    / "results"
    / "full_model"
    / "fusion_comparison"
    / "prott5"
    / aspect
    / "gated_bilinear"
    / "results.json"
    for aspect in ASPECTS
}


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_rows(path: Path) -> List[Tuple[str, str]]:
    rows = []
    with path.open("r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0], row[1]))
    return rows


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def save_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def verify_split_disjointness() -> Dict[str, Dict[str, int]]:
    summary = {}
    for aspect in ASPECTS:
        ids = {
            split: set(map(str, np.load(DATA_DIR / f"{aspect}_{split}_names.npy", allow_pickle=True).tolist()))
            for split in ["train", "valid", "test"]
        }
        counts = {
            "train_valid_overlap": len(ids["train"] & ids["valid"]),
            "train_test_overlap": len(ids["train"] & ids["test"]),
            "valid_test_overlap": len(ids["valid"] & ids["test"]),
        }
        if any(counts.values()):
            raise ValueError(f"Split overlap detected for {aspect}: {counts}")
        summary[aspect] = counts
    return summary


def verify_temporal_variant() -> Dict[str, int]:
    historical = load_protein_descriptions(HISTORICAL_OUTPUT_FILE)
    generated = load_protein_descriptions(TEMPORAL_PUNCT_FILE)
    mismatches = [
        protein_id
        for protein_id, text in generated.items()
        if clean_historical_description_artifacts(historical.get(protein_id, "")) != text
    ]
    if mismatches:
        sample = ", ".join(mismatches[:5])
        raise ValueError(f"Generated historical_punct_v1 text failed validation: {sample}")
    return {"historical_rows": len(historical), "generated_rows": len(generated), "mismatches": 0}


def encode_rows(rows: Sequence[Tuple[str, str]], output_dir: Path, batch_size: int = 16) -> int:
    if not rows:
        return 0
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    encoded = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(rows), batch_size), desc="Encoding mixed text cache"):
            batch = rows[start:start + batch_size]
            batch_ids = [protein_id for protein_id, _ in batch]
            batch_texts = [text or "None" for _, text in batch]
            encoding = tokenizer(
                batch_texts,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            hidden = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            ).last_hidden_state.cpu().numpy()
            for protein_id, embedding_np in zip(batch_ids, hidden):
                path = output_dir / f"{protein_id}.npy"
                temp_path = path.with_suffix(".tmp.npy")
                np.save(temp_path, embedding_np[None, ...])
                temp_path.rename(path)
                encoded += 1
    return encoded


def build_mixed_embedding_cache() -> Dict[str, int]:
    mixed_rows = load_rows(MIXED_OUTPUT_FILE)
    train_valid_ids = set(get_split_protein_ids(DATA_DIR, splits=["train", "valid"]))
    test_ids = set(get_split_protein_ids(DATA_DIR, splits=["test"]))

    linked_current = 0
    to_encode: List[Tuple[str, str]] = []

    MIXED_EMBED_DIR.mkdir(parents=True, exist_ok=True)

    for protein_id, text in mixed_rows:
        out_path = MIXED_EMBED_DIR / f"{protein_id}.npy"
        if out_path.exists():
            continue

        if protein_id in train_valid_ids:
            src = CURRENT_EMBED_DIR / f"{protein_id}.npy"
            if not src.exists():
                raise FileNotFoundError(f"Missing current embedding for train/valid protein {protein_id}")
            safe_link_or_copy(src, out_path)
            linked_current += 1
            continue

        if protein_id not in test_ids:
            raise ValueError(f"{protein_id} was not in train/valid or test splits")
        to_encode.append((protein_id, text))

    encoded = encode_rows(to_encode, MIXED_EMBED_DIR)
    total = len(list(MIXED_EMBED_DIR.glob("*.npy")))
    expected_total = len(mixed_rows)
    if total != expected_total:
        raise ValueError(f"Mixed embedding cache coverage mismatch: {total} vs expected {expected_total}")

    metadata = {
        "mixed_embedding_dir": str(MIXED_EMBED_DIR),
        "current_embedding_dir": str(CURRENT_EMBED_DIR),
        "records": expected_total,
        "linked_current": linked_current,
        "encoded": encoded,
    }
    save_json(TEMPORAL_DIR / "mixed_embedding_metadata.json", metadata)
    return metadata


def prepare_text_bundle() -> Dict:
    materialize_current_text_source()
    hist_meta = build_historical_punct_v1_test_tsv()
    mixed_meta = build_mixed_temporal_tsv()
    variant_check = verify_temporal_variant()
    return {
        "current_tsv": str(CURRENT_OUTPUT_FILE),
        "historical_tsv": str(HISTORICAL_OUTPUT_FILE),
        "historical_punct_meta": hist_meta,
        "mixed_meta": mixed_meta,
        "variant_check": variant_check,
        "bundle_metadata_file": str(TEXT_BUNDLE_METADATA),
    }


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def create_eval_model(num_go_terms: int, use_late_fusion: bool) -> torch.nn.Module:
    return create_model(
        fusion_type="gated_bilinear",
        seq_dim=1024,
        text_dim=768,
        struct_dim=512,
        ppi_dim=512,
        hidden_dim=512,
        num_go_terms=num_go_terms,
        dropout=0.4,
        modality_dropout=0.1,
        use_late_fusion=use_late_fusion,
    )


def evaluate_current_checkpoint(aspect: str) -> Dict:
    current_results_path = CURRENT_RESULTS[aspect]
    if current_results_path.exists():
        current_results = load_json(current_results_path)
    else:
        current_results = {"config": {"use_late_fusion": True}, "seed": 42,
                           "best_val_fmax": 0.0, "best_epoch": 0, "total_epochs": 0}
    checkpoint_path = current_results_path.parent / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {aspect}: {checkpoint_path}")

    output_dir = EVAL_OUTPUT_ROOT / "fusion_comparison" / "prott5" / aspect / "gated_bilinear"
    results_file = output_dir / "results.json"
    if results_file.exists():
        return load_json(results_file)

    embedding_dirs = {
        "text": str(MIXED_EMBED_DIR),
        "prott5": str(DATA_DIR / "embedding_cache" / "prott5"),
        "esm": str(DATA_DIR / "embedding_cache" / "esm"),
        "struct": str(DATA_DIR / "embedding_cache" / "IF1"),
        "ppi": str(DATA_DIR / "embedding_cache" / "ppi"),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = create_eval_model(
        num_go_terms=train_dataset.labels.shape[1],
        use_late_fusion=bool(current_results["config"].get("use_late_fusion", True)),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss().to(device)
    test_metrics = evaluate(model, test_loader, criterion, device, compute_weight_stats_detailed=True)
    go_terms = json.loads((DATA_DIR / f"{aspect}_go_terms.json").read_text())
    ia_file = DATA_DIR / f"{aspect}_ia.txt"
    cafa_metrics = evaluate_with_cafa(
        model=model,
        loader=test_loader,
        device=device,
        protein_ids=test_dataset.protein_ids.tolist(),
        go_terms=go_terms,
        obo_file=str(OBO_FILE),
        output_dir=str(output_dir / "cafa_eval"),
        model_name="prott5_gated_bilinear_temporal",
        train_labels=train_dataset.labels,
        ia_file=str(ia_file) if ia_file.exists() else None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "aspect": aspect,
        "source_current_results": str(current_results_path),
        "source_checkpoint": str(checkpoint_path),
        "text_embedding_dir": str(MIXED_EMBED_DIR),
        "text_recipe_name": "historical_punct_v1",
        "seed": int(current_results["seed"]),
        "best_val_fmax": float(current_results["best_val_fmax"]),
        "best_epoch": int(current_results["best_epoch"]),
        "total_epochs": int(current_results["total_epochs"]),
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
    if "late_fusion_lambda" in test_metrics:
        result["late_fusion_lambda"] = float(test_metrics["late_fusion_lambda"])
    with results_file.open("w") as f:
        json.dump(result, f, indent=2)
    return result


def run_evaluation() -> List[Dict]:
    rows = []
    for aspect in ASPECTS:
        result = evaluate_current_checkpoint(aspect)
        expected = EXPECTED[aspect]
        delta_fmax = float(result["cafa_fmax"]) - expected["fmax"]
        delta_wfmax = float(result["cafa_wfmax"]) - expected["wfmax"]
        rows.append(
            {
                "aspect": aspect,
                "expected_fmax": expected["fmax"],
                "expected_wfmax": expected["wfmax"],
                "actual_fmax": float(result["cafa_fmax"]),
                "actual_wfmax": float(result["cafa_wfmax"]),
                "delta_fmax": delta_fmax,
                "delta_wfmax": delta_wfmax,
                "passed": abs(delta_fmax) <= 0.0005 and abs(delta_wfmax) <= 0.0005,
                "results_json": str(
                    EVAL_OUTPUT_ROOT / "fusion_comparison" / "prott5" / aspect / "gated_bilinear" / "results.json"
                ),
            }
        )
    return rows


def main() -> int:
    split_summary = verify_split_disjointness()
    if MIXED_EMBED_DIR.exists() and any(MIXED_EMBED_DIR.glob("*.npy")):
        n = sum(1 for _ in MIXED_EMBED_DIR.glob("*.npy"))
        print(f"Text embedding cache present ({n} files), skipping text bundle preparation.")
        text_meta = {"status": "cached", "mixed_embedding_dir": str(MIXED_EMBED_DIR), "files": n}
        cache_meta = text_meta
    else:
        text_meta = prepare_text_bundle()
        cache_meta = build_mixed_embedding_cache()
    result_rows = run_evaluation()

    if not all(row["passed"] for row in result_rows):
        save_csv(SUMMARY_CSV, result_rows)
        save_json(
            SUMMARY_JSON,
            {
                "split_summary": split_summary,
                "text_meta": text_meta,
                "cache_meta": cache_meta,
                "results": result_rows,
            },
        )
        failed = [row for row in result_rows if not row["passed"]]
        raise SystemExit(
            "Reproduction failed for: "
            + ", ".join(
                f"{row['aspect']} (Fmax {row['actual_fmax']:.3f}, WFmax {row['actual_wfmax']:.3f})"
                for row in failed
            )
        )

    save_csv(SUMMARY_CSV, result_rows)
    save_json(
        SUMMARY_JSON,
        {
            "split_summary": split_summary,
            "text_meta": text_meta,
            "cache_meta": cache_meta,
            "results": result_rows,
        },
    )
    print(f"Reproduced temporal results. Summary: {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
