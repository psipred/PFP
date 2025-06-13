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

def get_sequences_to_process(
    pdb_dir: str,
    output_dir: str,
    force_regenerate: bool = False
) -> List[Tuple[str, str, str]]:
    """
    Get list of sequences that need processing.
    
    Returns:
        List of (protein_id, sequence, pdb_path) tuples
    """
    pdb_processor = PDBProcessor(pdb_dir)
    output_path = Path(output_dir)
    sequences_to_process = []
    
    # Get all PDB files
    pdb_files = list(Path(pdb_dir).glob("*.pdb"))
    logger.info(f"Found {len(pdb_files)} PDB files")
    
    for pdb_file in tqdm(pdb_files, desc="Checking which files need processing"):
        try:
            # Extract protein ID first
            protein_id = pdb_processor._extract_protein_id(pdb_file.name)
            output_file = output_path / f"{protein_id}.npy"
            
            # Skip if embedding already exists (unless force regenerate)
            if not force_regenerate and output_file.exists():
                continue
                
            # Only extract sequence if we need to process this file
            seq, coords, _ = pdb_processor.extract_sequence_and_coords(pdb_file)
            sequences_to_process.append((protein_id, seq, str(pdb_file)))
            
        except Exception as e:
            logger.warning(f"Failed to process {pdb_file.name}: {e}")
            continue
    
    logger.info(f"Need to process {len(sequences_to_process)} sequences")
    return sequences_to_process

def validate_sequence_for_esm(sequence: str, model_config) -> bool:
    """
    Validate if sequence can be processed by ESM model.
    
    Args:
        sequence: Protein sequence
        model_config: ESM model config
        
    Returns:
        True if sequence is valid
    """
    # Check length (leave room for special tokens)
    if len(sequence) > 1024:  # ESM typically handles up to 1024 residues
        logger.warning(f"Sequence too long: {len(sequence)} > 1024")
        return False
        
    # Check for invalid characters
    valid_aas = set('ACDEFGHIKLMNPQRSTVWYX')  # X for unknown
    invalid_chars = set(sequence) - valid_aas
    if invalid_chars:
        logger.warning(f"Invalid amino acids found: {invalid_chars}")
        return False
        
    return True

def generate_esm_embeddings_for_pdbs(
    pdb_dir: str = "/SAN/bioinf/PFP/embeddings/structure/pdb_files",
    output_dir: str = "/SAN/bioinf/PFP/embeddings/cafa5_small/esm_af",
    model_name: str = "facebook/esm1b_t33_650M_UR50S",
    batch_size: int = 4,  # Reduced batch size
    use_gpu: bool = True,
    mean_pool: bool = False,
    force_regenerate: bool = False,
    debug: bool = False
):
    """
    Generate ESM embeddings for sequences extracted from PDB files.
    """
    # Enable CUDA debugging if requested
    if debug and use_gpu:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        logging.info("CUDA debugging enabled")
    
    # Setup
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get sequences that need processing (efficient filtering)
    sequences_to_process = get_sequences_to_process(pdb_dir, output_dir, force_regenerate)
    
    if not sequences_to_process:
        logger.info("All embeddings already exist. Use force_regenerate=True to regenerate.")
        return
    
    # Load ESM model
    logger.info(f"Loading ESM model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Filter out invalid sequences
    valid_sequences = []
    for protein_id, sequence, pdb_path in sequences_to_process:
        if validate_sequence_for_esm(sequence, model.config):
            valid_sequences.append((protein_id, sequence, pdb_path))
        else:
            logger.warning(f"Skipping invalid sequence: {protein_id}")
    
    logger.info(f"Processing {len(valid_sequences)} valid sequences...")
    
    # Process in batches
    for i in tqdm(range(0, len(valid_sequences), batch_size), desc="Generating embeddings"):
        batch = valid_sequences[i:i + batch_size]
        batch_ids = [item[0] for item in batch]
        batch_seqs = [item[1] for item in batch]
        
        try:
            # Tokenize with proper max length
            encoded = tokenizer(
                batch_seqs,
                padding="longest",
                truncation=True,
                max_length=1026,  # 1024 residues + CLS + SEP
                return_tensors="pt",
                add_special_tokens=True
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Additional validation
            if input_ids.max() >= model.config.vocab_size:
                logger.error(f"Invalid token ID: {input_ids.max()} >= {model.config.vocab_size}")
                continue
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                
            # Save each sequence
            for idx, (protein_id, original_seq, _) in enumerate(batch):
                emb = embeddings[idx].cpu()
                seq_len = len(original_seq)
                
                # Extract residue embeddings (skip CLS token)
                residue_embeddings = emb[1:1+seq_len]
                
                if mean_pool:
                    saved_emb = residue_embeddings.mean(dim=0).numpy().astype(np.float32)
                else:
                    saved_emb = residue_embeddings.numpy().astype(np.float32)
                
                # Save
                output_file = output_path / f"{protein_id}.npy"
                np.save(
                    output_file,
                    {"name": protein_id, "embedding": saved_emb},
                    allow_pickle=True
                )
                
                logger.debug(f"Processed {protein_id}: seq_len={seq_len}, emb_shape={saved_emb.shape}")
                
        except RuntimeError as e:
            logger.error(f"Error processing batch {i//batch_size}: {e}")
            logger.error(f"Batch IDs: {batch_ids}")
            logger.error(f"Sequence lengths: {[len(seq) for seq in batch_seqs]}")
            
            # Process individually on error
            for protein_id, seq_str, _ in batch:
                try:
                    logger.info(f"Retrying {protein_id} individually...")
                    
                    # Validate sequence again
                    if not validate_sequence_for_esm(seq_str, model.config):
                        logger.warning(f"Skipping invalid sequence: {protein_id}")
                        continue
                    
                    # Process single sequence
                    single_encoded = tokenizer(
                        seq_str,
                        padding=False,
                        truncation=True,
                        max_length=1024,
                        return_tensors="pt",
                        add_special_tokens=True
                    )
                    
                    single_input_ids = single_encoded["input_ids"].to(device)
                    single_attention_mask = single_encoded["attention_mask"].to(device)
                    
                    # Additional validation
                    if single_input_ids.max() >= model.config.vocab_size:
                        logger.error(f"Invalid token ID for {protein_id}: {single_input_ids.max()}")
                        continue
                    
                    with torch.no_grad():
                        single_outputs = model(
                            input_ids=single_input_ids,
                            attention_mask=single_attention_mask
                        )
                        single_embedding = single_outputs.last_hidden_state
                    
                    # Save individual result
                    emb = single_embedding[0].cpu()
                    seq_len = len(seq_str)
                    residue_embeddings = emb[1:1+seq_len]
                    
                    if mean_pool:
                        saved_emb = residue_embeddings.mean(dim=0).numpy().astype(np.float32)
                    else:
                        saved_emb = residue_embeddings.numpy().astype(np.float32)
                    
                    output_file = output_path / f"{protein_id}.npy"
                    np.save(
                        output_file,
                        {"name": protein_id, "embedding": saved_emb},
                        allow_pickle=True
                    )
                    logger.info(f"Successfully processed {protein_id}: seq_len={seq_len}, emb_shape={saved_emb.shape}")
                    
                except Exception as e2:
                    logger.error(f"Failed to process {protein_id} individually: {e2}")
                    continue
    
    logger.info(f"Successfully generated embeddings for sequences")
    logger.info(f"Embeddings saved to: {output_dir}")


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
    parser.add_argument("--batch_size", type=int, default=4,
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