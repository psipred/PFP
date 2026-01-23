"""Extract ProtT5 protein sequence embeddings.

Usage:
    python extract_prott5_embeddings.py \
        --fasta_file <path_to_fasta> \
        --output_dir <path_to_output> \
        [--batch_size 2] \
        [--device cuda]

Output:
    Per-protein .npy files containing 1024-D embeddings (mean-pooled).
"""

import os
import re
import argparse
import numpy as np
import torch
from tqdm import tqdm
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel


def extract_prott5_embeddings(
    fasta_file: str,
    output_dir: str,
    model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
    batch_size: int = 2,
    max_length: int = 1024,
    device: str = "cuda"
):
    """
    Extract ProtT5 embeddings from protein sequences.

    Args:
        fasta_file: Path to FASTA file with protein sequences.
        output_dir: Directory to save .npy embedding files.
        model_name: HuggingFace model name for ProtT5.
        batch_size: Number of sequences per batch.
        max_length: Maximum sequence length (truncated).
        device: 'cuda' or 'cpu'.

    Output format:
        {protein_id}.npy: numpy array of shape (1024,)
    """
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)

    # Use half precision on GPU
    if device.type == 'cuda':
        model = model.half()
    else:
        model = model.float()

    model = model.to(device)
    model.eval()

    # Parse FASTA and filter already processed
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_id = record.id
        out_file = os.path.join(output_dir, f"{seq_id}.npy")
        if not os.path.isfile(out_file):
            sequences[seq_id] = str(record.seq)

    if not sequences:
        print(f"All embeddings already exist in '{output_dir}'. Nothing to do.")
        return

    print(f"Processing {len(sequences)} protein sequences...")

    # Format sequence for ProtT5 (space-separated amino acids, replace rare AAs)
    def format_sequence(seq):
        seq = re.sub(r"[UZOB]", "X", seq)
        return " ".join(list(seq))

    # Create batches
    def create_batches(seqs, batch_size):
        keys = list(seqs.keys())
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i+batch_size]
            batch_seqs = [format_sequence(seqs[k]) for k in batch_keys]
            yield batch_keys, batch_seqs

    # Process batches
    with torch.no_grad():
        for seq_ids, batch_seqs in tqdm(
            create_batches(sequences, batch_size),
            total=(len(sequences) + batch_size - 1) // batch_size,
            desc="Extracting ProtT5 embeddings"
        ):
            # Tokenize
            tokenized = tokenizer(
                batch_seqs,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt"
            )

            input_ids = tokenized["input_ids"][:, :max_length].to(device)
            attention_mask = tokenized["attention_mask"][:, :max_length].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Mean pool over sequence length
            embeddings = outputs.last_hidden_state  # [batch, seq_len, 1024]

            # Mask-aware mean pooling
            for i, seq_id in enumerate(seq_ids):
                mask = attention_mask[i].unsqueeze(-1).float()
                seq_len = mask.sum()
                if seq_len > 0:
                    embedding = (embeddings[i] * mask).sum(dim=0) / seq_len
                else:
                    embedding = embeddings[i].mean(dim=0)

                # Save as numpy array
                emb_np = embedding.cpu().float().numpy()
                np.save(os.path.join(output_dir, f"{seq_id}.npy"), emb_np)

    print(f"Done! Embeddings saved in '{output_dir}'")


def main():
    parser = argparse.ArgumentParser(description="Extract ProtT5 embeddings")
    parser.add_argument("--fasta_file", type=str, required=True,
                        help="Path to FASTA file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--model_name", type=str,
                        default="Rostlab/prot_t5_xl_half_uniref50-enc",
                        help="HuggingFace model name")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    extract_prott5_embeddings(
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )


if __name__ == "__main__":
    main()
