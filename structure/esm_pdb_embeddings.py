#!/usr/bin/env python3
"""
Generate ESM embeddings for protein sequences extracted from PDB files.
This ensures alignment between structure and sequence embeddings.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdb_graph_utils import PDBProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_esm_embeddings_for_pdbs(
    pdb_dir: str = "/SAN/bioinf/PFP/embeddings/structure/pdb_files",
    output_dir: str = "/SAN/bioinf/PFP/embeddings/cafa5_small/esm_af",
    model_name: str = "facebook/esm1b_t33_650M_UR50S",
    batch_size: int = 8,
    use_gpu: bool = True,
    mean_pool: bool = False,  # Keep per-residue for node features
    force_regenerate: bool = False,
    debug: bool = False
):
    """
    Generate ESM embeddings for sequences extracted from PDB files.
    
    Args:
        pdb_dir: Directory containing PDB files
        output_dir: Where to save embeddings
        model_name: ESM model to use
        batch_size: Batch size for inference
        use_gpu: Use GPU if available
        mean_pool: If True, average over sequence length
        force_regenerate: Regenerate even if files exist
        debug: Enable debug mode for CUDA errors
    """
    # Enable CUDA debugging if requested
    if debug and use_gpu:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        logging.info("CUDA debugging enabled")
    
    # Setup
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Clear GPU cache to avoid memory issues
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load ESM model
    logger.info(f"Loading ESM model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Log model info
    logger.info(f"Model vocab size: {model.config.vocab_size}")
    logger.info(f"Model max position embeddings: {model.config.max_position_embeddings}")
    
    # Process PDB files to extract sequences
    logger.info("Extracting sequences from PDB files...")
    pdb_processor = PDBProcessor(pdb_dir)
    sequence_data = pdb_processor.process_all_pdbs()
    
    # Filter sequences that need processing
    sequences_to_process = []
    for protein_id, (sequence, _) in sequence_data.items():
        output_file = output_path / f"{protein_id}.npy"
        if force_regenerate or not output_file.exists():
            sequences_to_process.append((protein_id, sequence))
        
    if not sequences_to_process:
        logger.info("All embeddings already exist. Use force_regenerate=True to regenerate.")
        return
        
    logger.info(f"Generating embeddings for {len(sequences_to_process)} sequences...")
    
    # Process in batches
    for i in tqdm(range(0, len(sequences_to_process), batch_size), desc="Generating embeddings"):
        batch = sequences_to_process[i:i + batch_size]
        batch_ids = [item[0] for item in batch]
        batch_seqs = [item[1] for item in batch]
        
        try:
            # Tokenize - set max_length to accommodate sequence + special tokens
            encoded = tokenizer(
                batch_seqs,
                padding="longest",
                truncation=True,
                max_length=1026,  # 1024 residues + CLS + SEP tokens
                return_tensors="pt",
                add_special_tokens=True
            )
            
            # Check for issues before sending to GPU
            if encoded["input_ids"].shape[1] > 1026:
                logger.warning(f"Sequence too long: {encoded['input_ids'].shape[1]} tokens")
                continue
            
            # Move to device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Verify token IDs are in valid range
            if input_ids.max() >= model.config.vocab_size:
                logger.error(f"Invalid token ID found: {input_ids.max()} >= {model.config.vocab_size}")
                continue
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs[0]  # (batch_size, seq_len, hidden_dim)
                
        except RuntimeError as e:
            logger.error(f"Error processing batch {i//batch_size}: {e}")
            logger.error(f"Batch IDs: {batch_ids}")
            logger.error(f"Sequence lengths: {[len(seq) for seq in batch_seqs]}")
            
            # Try processing sequences individually
            for j, (seq_id, seq_str) in enumerate(batch):
                try:
                    logger.info(f"Retrying {seq_id} individually...")
                    
                    # Process single sequence - accommodate full sequence + special tokens
                    single_encoded = tokenizer(
                        seq_str,
                        padding=False,
                        truncation=True,
                        max_length=1026,  # 1024 residues + CLS + SEP tokens
                        return_tensors="pt",
                        add_special_tokens=True
                    )
                    
                    single_input_ids = single_encoded["input_ids"].to(device)
                    single_attention_mask = single_encoded["attention_mask"].to(device)
                    
                    with torch.no_grad():
                        single_outputs = model(
                            input_ids=single_input_ids,
                            attention_mask=single_attention_mask
                        )
                        single_embedding = single_outputs[0]
                    
                    # Save individual result - FIXED
                    mask = single_attention_mask[0].bool()
                    emb = single_embedding[0].cpu()
                    
                    # Find actual sequence tokens (excluding CLS and SEP)
                    # CLS is at position 0, SEP is at the last non-padded position
                    seq_len = len(seq_str)
                    residue_embeddings = emb[1:1+seq_len]  # Skip CLS, take exactly seq_len tokens
                    
                    if mean_pool:
                        saved_emb = residue_embeddings.mean(dim=0).numpy().astype(np.float32)
                    else:
                        saved_emb = residue_embeddings.numpy().astype(np.float32)
                    
                    output_file = output_path / f"{seq_id}.npy"
                    np.save(
                        output_file,
                        {"name": seq_id, "embedding": saved_emb},
                        allow_pickle=True
                    )
                    logger.info(f"Successfully processed {seq_id}: seq_len={seq_len}, emb_shape={saved_emb.shape}")
                    
                except Exception as e2:
                    logger.error(f"Failed to process {seq_id} even individually: {e2}")
                    continue
            
            continue
        
        # Save each sequence - FIXED
        for idx, (protein_id, original_seq) in enumerate(zip(batch_ids, batch_seqs)):
            emb = embeddings[idx].cpu()
            
            # Get actual sequence length and extract corresponding embeddings
            seq_len = len(original_seq)
            residue_embeddings = emb[1:1+seq_len]  # Skip CLS, take exactly seq_len tokens
            
            if mean_pool:
                saved_emb = residue_embeddings.mean(dim=0).numpy().astype(np.float32)
            else:
                saved_emb = residue_embeddings.numpy().astype(np.float32)
            
            # Save in same format as existing embeddings
            output_file = output_path / f"{protein_id}.npy"
            np.save(
                output_file,
                {"name": protein_id, "embedding": saved_emb},
                allow_pickle=True
            )
            
            logger.debug(f"Processed {protein_id}: seq_len={seq_len}, emb_shape={saved_emb.shape}")
    
    logger.info(f"Successfully generated embeddings for {len(sequences_to_process)} sequences")
    logger.info(f"Embeddings saved to: {output_dir}")
    
    # Summary statistics
    total_pdbs = len(sequence_data)
    processed = len(sequences_to_process)
    skipped = total_pdbs - processed
    
    logger.info(f"\nSummary:")
    logger.info(f"  Total PDB files: {total_pdbs}")
    logger.info(f"  Newly processed: {processed}")
    logger.info(f"  Skipped (existing): {skipped}")


def validate_embeddings(
    pdb_dir: str = "/SAN/bioinf/PFP/embeddings/structure/pdb_files",
    embedding_dir: str = "/SAN/bioinf/PFP/embeddings/cafa5_small/esm_af"
):
    """Validate that embeddings match PDB sequences."""
    pdb_processor = PDBProcessor(pdb_dir)
    embedding_path = Path(embedding_dir)
    
    logger.info("Validating embeddings...")
    
    # Get a few samples
    pdb_files = list(Path(pdb_dir).glob("*.pdb"))[:50]
    
    mismatches = []
    for pdb_file in pdb_files:
        try:
            seq, coords, protein_id = pdb_processor.extract_sequence_and_coords(pdb_file)
            
            # Load embedding
            emb_file = embedding_path / f"{protein_id}.npy"
            if not emb_file.exists():
                logger.warning(f"Missing embedding for {protein_id}")
                continue
                
            data = np.load(emb_file, allow_pickle=True).item()
            embeddings = data['embedding']
            
            # Check dimensions
            if embeddings.ndim == 2:  # Per-residue embeddings
                if embeddings.shape[0] != len(seq):
                    mismatches.append((protein_id, len(seq), embeddings.shape[0]))
            
            logger.info(f"✓ {protein_id}: seq_len={len(seq)}, emb_shape={embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error validating {pdb_file}: {e}")
    
    if mismatches:
        logger.warning(f"\nFound {len(mismatches)} length mismatches:")
        for pid, seq_len, emb_len in mismatches:
            logger.warning(f"  {pid}: seq={seq_len}, emb={emb_len}")
    else:
        logger.info("\nAll validated embeddings match their sequences!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ESM embeddings for PDB sequences")
    parser.add_argument("--pdb_dir", type=str, 
                       default="/SAN/bioinf/PFP/embeddings/structure/pdb_files",
                       help="Directory containing PDB files")
    parser.add_argument("--output_dir", type=str,
                       default="/SAN/bioinf/PFP/embeddings/cafa5_small/esm_af",
                       help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for ESM inference")
    parser.add_argument("--force", action="store_true",
                       help="Force regeneration of existing embeddings")
    parser.add_argument("--validate", action="store_true",
                       help="Validate embeddings after generation")
    parser.add_argument("--mean_pool", action="store_true",
                       help="Average embeddings over sequence length")
    parser.add_argument("--debug", action="store_true",
                       help="Enable CUDA debugging")
    
    args = parser.parse_args()
    
    # Generate embeddings
    generate_esm_embeddings_for_pdbs(
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        force_regenerate=args.force,
        mean_pool=args.mean_pool,
        debug=args.debug
    )
    
    # Validate if requested
    if args.validate:
        validate_embeddings(args.pdb_dir, args.output_dir)