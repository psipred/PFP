"""Extract protein structure embeddings using ESM-IF1.

ESM-IF1 is trained for inverse folding but provides high-quality structure
embeddings from backbone coordinates.

Usage:
    python extract_esm_if1_embeddings.py \
        --pdb_dir <path_to_pdbs> \
        --output_dir <path_to_output> \
        [--pooling mean] \
        [--device cuda] \
        [--chain A]

Output:
    Per-protein .npy files containing 512-D structure embeddings.
"""

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")


def extract_embeddings(
    pdb_dir: str,
    output_dir: str,
    pooling: str = "mean",
    device: str = "cuda",
    chain_id: str = None
):
    """
    Extract ESM-IF1 structure embeddings from PDB files.

    Args:
        pdb_dir: Directory containing PDB files.
        output_dir: Directory to save .npy embedding files.
        pooling: Pooling strategy ('mean', 'max', or 'cls').
        device: 'cuda' or 'cpu'.
        chain_id: Single chain ID (e.g. 'A'), or None to auto-select.

    Output format:
        {protein_id}.npy: numpy array of shape (512,)
    """
    # Import ESM
    try:
        import esm
        import esm.inverse_folding
    except ImportError as e:
        print("ERROR: Could not import `esm` (fair-esm).")
        print("  Try: pip install fair-esm")
        print(f"  Original error: {e}")
        return

    pdb_dir = Path(pdb_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get PDB files to process (skip existing)
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    existing = {f.stem for f in output_dir.glob("*.npy")}
    pdb_files = [f for f in pdb_files if f.stem not in existing]

    print(f"Found {len(pdb_files)} PDB files to process")
    if len(pdb_files) == 0:
        print("All embeddings already generated!")
        return

    # Load model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Loading ESM-IF1 model on {device}...")

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device).eval()

    print("Model loaded.")
    print(f"  Embedding dimension: 512")
    print(f"  Pooling: {pooling}")
    print(f"  Chain: {chain_id if chain_id else 'auto / first chain'}\n")

    failed = []

    with torch.no_grad():
        for pdb_file in tqdm(pdb_files, desc="Extracting"):
            protein_id = pdb_file.stem

            try:
                # Load structure
                if chain_id is not None:
                    structure = esm.inverse_folding.util.load_structure(
                        str(pdb_file), chain_id
                    )
                else:
                    structure = esm.inverse_folding.util.load_structure(
                        str(pdb_file)
                    )

                # Extract backbone coordinates and sequence
                coords, seq = esm.inverse_folding.util.extract_coords_from_structure(
                    structure
                )

                if coords is None or len(coords) == 0:
                    raise ValueError("No backbone coordinates found")

                # Get encoder output (L x 512)
                rep = esm.inverse_folding.util.get_encoder_output(
                    model, alphabet, coords
                )

                # Pool to protein-level embedding
                if pooling == "mean":
                    embedding = rep.mean(dim=0)
                elif pooling == "max":
                    embedding = rep.max(dim=0)[0]
                elif pooling == "cls":
                    embedding = rep[0]
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")

                # Save
                embedding_np = embedding.detach().cpu().numpy()
                np.save(output_dir / f"{protein_id}.npy", embedding_np)

            except Exception as e:
                failed.append((protein_id, str(e)))

    print("\nSummary:")
    print(f"  Saved:  {len(pdb_files) - len(failed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Output dir: {output_dir}")

    if failed:
        print("\nExamples of failed proteins:")
        for pid, err in failed[:10]:
            print(f"  {pid}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-IF1 embeddings")
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Directory with PDB files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "cls"],
                        help="Pooling strategy")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--chain", type=str, default=None,
                        help="Chain ID to use (e.g. 'A')")

    args = parser.parse_args()

    extract_embeddings(
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        pooling=args.pooling,
        device=args.device,
        chain_id=args.chain
    )


if __name__ == "__main__":
    main()
