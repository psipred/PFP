"""Reproduce Table 2: Modality Contribution (retrain subsets).

For each variant (seq_only, seq_text, seq_struct, seq_ppi):
1. Train gated_bilinear with appropriate modalities disabled via empty dirs
2. Run CAFA evaluation with IA weights
3. Output summary CSV matching Table 2
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from train import train_fusion_model

ASPECTS = ["BPO", "CCO", "MFO"]
DATA_DIR = PROJECT_DIR / "data"
OBO_FILE = DATA_DIR / "go.obo"
IA_DIR = DATA_DIR
TEXT_EMBED_DIR = DATA_DIR / "embedding_cache" / "exp_text_embeddings_temporal"
OUTPUT_ROOT = PROJECT_DIR / "results" / "modality_contribution"
SUMMARY_CSV = OUTPUT_ROOT / "modality_contribution_summary.csv"
SUMMARY_JSON = OUTPUT_ROOT / "modality_contribution_summary.json"

VARIANTS = [
    {
        "name": "seq_only",
        "enabled_modalities": {"prott5"},
    },
    {
        "name": "seq_text",
        "enabled_modalities": {"prott5", "text"},
    },
    {
        "name": "seq_struct",
        "enabled_modalities": {"prott5", "struct"},
    },
    {
        "name": "seq_ppi",
        "enabled_modalities": {"prott5", "ppi"},
    },
]

EXPECTED = {
    "seq_only": {
        "BPO": {"wfmax": 0.484}, "MFO": {"wfmax": 0.573}, "CCO": {"wfmax": 0.546},
    },
    "seq_text": {
        "BPO": {"wfmax": 0.497}, "MFO": {"wfmax": 0.594}, "CCO": {"wfmax": 0.557},
    },
    "seq_struct": {
        "BPO": {"wfmax": 0.487}, "MFO": {"wfmax": 0.586}, "CCO": {"wfmax": 0.552},
    },
    "seq_ppi": {
        "BPO": {"wfmax": 0.505}, "MFO": {"wfmax": 0.594}, "CCO": {"wfmax": 0.558},
    },
}


def canonical_embedding_dirs() -> Dict[str, str]:
    return {
        "text": str(TEXT_EMBED_DIR),
        "prott5": str(DATA_DIR / "embedding_cache" / "prott5"),
        "esm": str(DATA_DIR / "embedding_cache" / "esm"),
        "struct": str(DATA_DIR / "embedding_cache" / "IF1"),
        "ppi": str(DATA_DIR / "embedding_cache" / "ppi"),
    }


def make_variant_dirs(variant_name: str, enabled: set) -> Dict[str, str]:
    """Create embedding dirs with empty dirs for disabled modalities."""
    full = canonical_embedding_dirs()
    result = {}
    for modality, path in full.items():
        if modality in enabled:
            result[modality] = path
        else:
            empty_dir = OUTPUT_ROOT / variant_name / f"_empty_{modality}"
            empty_dir.mkdir(parents=True, exist_ok=True)
            result[modality] = str(empty_dir)
    return result


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


def train_variant(variant: Dict, aspect: str) -> Dict:
    output_base = OUTPUT_ROOT / variant["name"]
    results_file = (
        output_base / "fusion_comparison" / "prott5" / aspect / "gated_bilinear" / "results.json"
    )
    if results_file.exists():
        existing = load_json(results_file)
        if existing.get("cafa_fmax") is not None and existing.get("cafa_wfmax") is not None:
            return existing

    embedding_dirs = make_variant_dirs(variant["name"], variant["enabled_modalities"])

    result = train_fusion_model(
        seq_model="prott5",
        aspect=aspect,
        fusion_type="gated_bilinear",
        data_dir=str(DATA_DIR),
        embedding_dirs=embedding_dirs,
        obo_file=str(OBO_FILE),
        modality_dropout=0.1,
        output_base=str(output_base),
        use_late_fusion=True,
        late_output_mode="hybrid",
        aux_loss_weight=0.8,
        num_workers=0,
        ia_file=str(IA_DIR / f"{aspect}_ia.txt") if (IA_DIR / f"{aspect}_ia.txt").exists() else None,
        seed=42,
    )
    return result


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict] = []

    for variant in VARIANTS:
        for aspect in ASPECTS:
            print(f"\n{'='*60}")
            print(f"Training {variant['name']} / {aspect}")
            print(f"{'='*60}")

            result = train_variant(variant, aspect)
            expected = EXPECTED.get(variant["name"], {}).get(aspect, {})

            row = {
                "variant": variant["name"],
                "aspect": aspect,
                "cafa_fmax": float(result.get("cafa_fmax", 0)),
                "cafa_wfmax": float(result.get("cafa_wfmax", 0)),
                "expected_wfmax": expected.get("wfmax", ""),
                "delta_wfmax": float(result.get("cafa_wfmax", 0)) - expected.get("wfmax", 0)
                if expected.get("wfmax") is not None else "",
            }
            summary_rows.append(row)

    save_csv(SUMMARY_CSV, summary_rows)
    save_json(SUMMARY_JSON, {"rows": summary_rows})

    print(f"\nModality contribution results saved to {SUMMARY_CSV}")

    # Check tolerances
    failed = []
    for row in summary_rows:
        if row["delta_wfmax"] != "" and abs(row["delta_wfmax"]) > 0.001:
            failed.append(f"{row['variant']}/{row['aspect']}: wFmax={row['cafa_wfmax']:.3f} (expected {row['expected_wfmax']:.3f})")

    if failed:
        print("\nWARNING: The following results exceeded tolerance:")
        for f in failed:
            print(f"  {f}")
        return 1

    print("\nAll results within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
