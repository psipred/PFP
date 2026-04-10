#!/usr/bin/env python3
"""
Extract PPI (Protein-Protein Interaction) embeddings from STRING database.

STRING provides pre-computed network embeddings (512-D) that capture
protein interaction network topology.

This script handles two types of protein IDs:
- UniProt IDs (train/valid sets): Map directly to STRING IDs
- CAFA3 IDs (test set): Map CAFA3 -> Entry Name -> UniProt -> STRING
  using the comprehensive CAFA assessment tool mappings

Usage:
    python scripts/extract_ppi_embeddings.py

    # Limit for testing
    python scripts/extract_ppi_embeddings.py --limit 100

Required external files (configured below):
    - STRING h5 file: protein.network.embeddings.v12.0.h5
    - STRING alias file: protein.aliases.v12.0.txt
    - CAFA assessment tool: for CAFA ID mappings

Output:
    data/embedding_cache/ppi/{protein_id}.npy (512-D vectors)
"""

import os
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Update these paths for your system
# ============================================================================
STRING_H5_FILE = os.environ.get("STRING_H5_FILE", "protein.network.embeddings.v12.0.h5")
STRING_ALIAS_FILE = os.environ.get("STRING_ALIAS_FILE", "protein.aliases.v12.0.txt")
CAFA_ASSESSMENT_DIR = os.environ.get("CAFA_ASSESSMENT_DIR", "CAFA_assessment_tool")
CAFA3_ID_MAPPING = os.environ.get("CAFA3_ID_MAPPING", "cafa3_id_mapping.json")
DATA_DIR = "./data"
OUTPUT_DIR = "./data/embedding_cache/ppi"
# ============================================================================


def read_target_mapping(taxon, target_folder):
    """Read CAFA ID -> UniProt entry name mapping."""
    # Taxons with sp_species.{taxon}.map format
    sp_taxons = ['85962', '83333', '10116', '7955', '273057', '160488',
                 '170187', '224308', '243273', '9606', '243232', '44689',
                 '559292', '284812', '3702', '8355', '10090', '223283',
                 '99287', '321314']
    # Taxons with mapping.{taxon}.map format
    mapping_taxons = ['208963', '7227', '237561']

    all_taxons = sp_taxons + mapping_taxons

    targetdict = {}
    if taxon not in all_taxons:
        return targetdict

    if taxon in mapping_taxons:
        filename = f'mapping.{taxon}.map'
    else:
        filename = f'sp_species.{taxon}.map'

    filepath = Path(target_folder) / filename
    if not filepath.exists():
        return targetdict

    with open(filepath, 'r') as handle:
        for line in handle:
            fields = line.strip().split('\t')
            if len(fields) >= 2:
                cafa_id = fields[0]
                entry_name = fields[1]
                targetdict[cafa_id] = entry_name

    return targetdict


def read_uniprot_mapping(taxon, uniprot_folder):
    """Read UniProt entry name -> UniProt accession mapping."""
    # All supported taxons
    all_taxons = ['85962', '83333', '10116', '7955', '273057', '160488',
                  '170187', '224308', '243273', '9606', '243232', '44689',
                  '559292', '284812', '3702', '8355', '10090', '223283',
                  '99287', '321314', '208963', '7227', '237561']

    # Taxons with .map format (3-column: accession, type, entry_name)
    map_taxons = ['10090', '10116', '284812', '3702', '44689', '559292',
                  '7227', '7955', '83333', '9606']

    uniprotdict = {}

    if taxon not in all_taxons:
        return uniprotdict

    if taxon in map_taxons:
        filename = f'uniprot_ac_to_id_{taxon}.map'
        mapping = True
    else:
        filename = f'uniprot_ac_to_id_{taxon}.tab'
        mapping = False

    filepath = Path(uniprot_folder) / filename
    if not filepath.exists():
        return uniprotdict

    with open(filepath, 'r') as f:
        f.readline()  # Skip header if exists
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            accession = parts[0]
            entry_name = parts[2] if mapping and len(parts) > 2 else parts[1]

            if entry_name not in uniprotdict:
                uniprotdict[entry_name] = accession

    return uniprotdict


def build_cafa_to_uniprot_mapping(cafa_assessment_dir):
    """Build CAFA ID -> UniProt Accession mapping using CAFA assessment tool."""
    logger.info("Building CAFA -> UniProt mapping from CAFA assessment tool...")

    # All supported taxons (sp_species + mapping formats)
    all_taxons = ['85962', '83333', '10116', '7955', '273057', '160488',
                  '170187', '224308', '243273', '9606', '243232', '44689',
                  '559292', '284812', '3702', '8355', '10090', '223283',
                  '99287', '321314', '208963', '7227', '237561']

    target_folder = Path(cafa_assessment_dir) / "ID_conversion" / "CAFA_mapping"
    uniprot_folder = Path(cafa_assessment_dir) / "ID_conversion" / "uniprot_mapping"

    cafa_to_uniprot = {}  # cafa_id -> uniprot_accession

    for taxon in all_taxons:
        target_dict = read_target_mapping(taxon, target_folder)
        uniprot_dict = read_uniprot_mapping(taxon, uniprot_folder)

        mapped = 0
        for cafa_id, entry_name in target_dict.items():
            if entry_name in uniprot_dict:
                accession = uniprot_dict[entry_name]
                cafa_to_uniprot[cafa_id] = accession
                mapped += 1

        if mapped > 0:
            logger.info(f"  Taxon {taxon}: {mapped} mappings")

    logger.info(f"Total CAFA -> UniProt mappings: {len(cafa_to_uniprot)}")
    return cafa_to_uniprot


def build_uniprot_alternative_mapping(cafa_assessment_dir):
    """Build UniProt alternative ID mappings via entry names.

    This helps map outdated/alternative UniProt IDs to current STRING-recognized ones.
    Returns:
    - uniprot_to_entry: UniProt accession -> entry name
    - entry_to_uniprots: entry name -> list of all known accessions
    """
    logger.info("Building UniProt alternative ID mapping via entry names...")

    # All supported taxons
    all_taxons = ['85962', '83333', '10116', '7955', '273057', '160488',
                  '170187', '224308', '243273', '9606', '243232', '44689',
                  '559292', '284812', '3702', '8355', '10090', '223283',
                  '99287', '321314', '208963', '7227', '237561']

    # Taxons with .map format (3-column: accession, type, entry_name)
    map_taxons = ['10090', '10116', '284812', '3702', '44689', '559292',
                  '7227', '7955', '83333', '9606']

    uniprot_folder = Path(cafa_assessment_dir) / "ID_conversion" / "uniprot_mapping"

    uniprot_to_entry = {}  # accession -> entry_name
    entry_to_uniprots = {}  # entry_name -> list of accessions

    for taxon in all_taxons:
        if taxon in map_taxons:
            filename = f'uniprot_ac_to_id_{taxon}.map'
            mapping = True
        else:
            filename = f'uniprot_ac_to_id_{taxon}.tab'
            mapping = False

        filepath = uniprot_folder / filename
        if not filepath.exists():
            continue

        with open(filepath, 'r') as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                accession = parts[0]
                entry_name = parts[2] if mapping and len(parts) > 2 else parts[1]

                # Map accession to entry name
                uniprot_to_entry[accession] = entry_name

                # Track all accessions for each entry
                if entry_name not in entry_to_uniprots:
                    entry_to_uniprots[entry_name] = []
                if accession not in entry_to_uniprots[entry_name]:
                    entry_to_uniprots[entry_name].append(accession)

    logger.info(f"UniProt -> Entry: {len(uniprot_to_entry)} mappings")
    logger.info(f"Entry -> UniProt(s): {len(entry_to_uniprots)} entries")
    return uniprot_to_entry, entry_to_uniprots


def build_alias_to_string_mapping(alias_file: Path) -> dict:
    """Build ID -> STRING mapping from alias file.

    Includes UniProt IDs and FlyBase gene IDs for comprehensive coverage.
    """
    logger.info("Building ID -> STRING mapping...")

    alias_to_string = {}

    # Sources to include for mapping
    valid_sources = [
        'UniProt_AC',           # UniProt accession (e.g., P12345)
        'UniProt_ID',           # UniProt entry name (e.g., 1433B_HUMAN)
        'UniProt_DR_FlyBase',   # FlyBase cross-reference (e.g., FBgn0030647)
        'Ensembl_gene',         # Ensembl gene ID (includes FBgn for Drosophila)
        'Ensembl_flybase_gene_id',  # FlyBase gene ID directly
    ]

    with open(alias_file, 'r') as f:
        next(f)  # Skip header
        for line in tqdm(f, desc="Reading aliases"):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                string_id = parts[0]
                alias = parts[1]
                source = parts[2]

                # Use valid sources for mapping
                if source in valid_sources:
                    if alias not in alias_to_string:
                        alias_to_string[alias] = string_id

    logger.info(f"ID -> STRING: {len(alias_to_string)} mappings")
    return alias_to_string


def is_cafa3_id(protein_id: str) -> bool:
    """Check if protein ID is a CAFA3 ID (starts with T and is long)."""
    return protein_id.startswith('T') and len(protein_id) > 10


def load_cafa3_id_mapping(mapping_file: Path) -> dict:
    """Load CAFA3 ID -> entry name/gene ID mapping from JSON file."""
    import json
    logger.info(f"Loading CAFA3 ID mapping from {mapping_file}")

    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    logger.info(f"Loaded {len(mapping)} CAFA3 ID mappings")
    return mapping


def get_needed_string_ids(data_dir: Path, cafa_to_uniprot: dict,
                          cafa_to_id: dict, alias_to_string: dict,
                          uniprot_to_entry: dict, entry_to_uniprots: dict,
                          limit: int = None) -> dict:
    """Get STRING IDs needed based on protein data files.

    Handles two cases:
    - UniProt IDs (train/valid): Map directly to STRING, or via entry name alternatives
    - CAFA3 IDs (test): Try multiple mapping paths:
      1. CAFA3 -> UniProt (from CAFA assessment) -> STRING
      2. CAFA3 -> Gene ID (e.g., FBgn for Drosophila) -> STRING
    """
    logger.info("Finding required STRING IDs...")

    # Get all protein IDs from data splits
    all_protein_ids = set()
    for aspect in ['BPO', 'CCO', 'MFO']:
        for split in ['train', 'valid', 'test']:
            names_file = data_dir / f"{aspect}_{split}_names.npy"
            if names_file.exists():
                proteins = np.load(names_file, allow_pickle=True)
                all_protein_ids.update(proteins)

    logger.info(f"Total unique protein IDs: {len(all_protein_ids)}")

    if limit:
        all_protein_ids = set(list(all_protein_ids)[:limit])
        logger.info(f"Limited to {len(all_protein_ids)} proteins")

    # Separate CAFA3 IDs and UniProt IDs
    cafa3_ids = {p for p in all_protein_ids if is_cafa3_id(p)}
    uniprot_ids = all_protein_ids - cafa3_ids

    logger.info(f"UniProt IDs (train/valid): {len(uniprot_ids)}")
    logger.info(f"CAFA3 IDs (test): {len(cafa3_ids)}")

    # Map to STRING IDs
    needed_string_ids = {}  # string_id -> set of original protein_ids
    mapped_uniprot = 0
    mapped_uniprot_via_entry = 0
    mapped_cafa3 = 0
    mapped_cafa3_via_geneid = 0

    # Process UniProt IDs with multiple strategies
    for uniprot_id in tqdm(uniprot_ids, desc="Mapping UniProt->STRING"):
        string_id = None

        # Strategy 1: Direct lookup
        if uniprot_id in alias_to_string:
            string_id = alias_to_string[uniprot_id]
        else:
            # Strategy 2: Try without isoform suffix (P12345-2 -> P12345)
            base_id = uniprot_id.split('-')[0]
            if base_id in alias_to_string:
                string_id = alias_to_string[base_id]

        # Strategy 3: Map via entry name -> alternative UniProt IDs
        if string_id is None and uniprot_id in uniprot_to_entry:
            entry_name = uniprot_to_entry[uniprot_id]
            # Try entry name directly (STRING has UniProt_ID source)
            if entry_name in alias_to_string:
                string_id = alias_to_string[entry_name]
                mapped_uniprot_via_entry += 1
            else:
                # Try all known accessions for this entry
                if entry_name in entry_to_uniprots:
                    for alt_acc in entry_to_uniprots[entry_name]:
                        if alt_acc in alias_to_string:
                            string_id = alias_to_string[alt_acc]
                            mapped_uniprot_via_entry += 1
                            break

        if string_id:
            if string_id not in needed_string_ids:
                needed_string_ids[string_id] = set()
            needed_string_ids[string_id].add(uniprot_id)
            mapped_uniprot += 1

    # Process CAFA3 IDs with multiple fallback strategies
    for cafa3_id in tqdm(cafa3_ids, desc="Mapping CAFA3->STRING"):
        string_id = None

        # Strategy 1: CAFA3 -> UniProt (from CAFA assessment tool) -> STRING
        if cafa3_id in cafa_to_uniprot:
            uniprot_id = cafa_to_uniprot[cafa3_id]
            if uniprot_id in alias_to_string:
                string_id = alias_to_string[uniprot_id]
            else:
                base_id = uniprot_id.split('-')[0]
                if base_id in alias_to_string:
                    string_id = alias_to_string[base_id]

            # Try via entry name if direct lookup fails
            if string_id is None and uniprot_id in uniprot_to_entry:
                entry_name = uniprot_to_entry[uniprot_id]
                if entry_name in alias_to_string:
                    string_id = alias_to_string[entry_name]
                elif entry_name in entry_to_uniprots:
                    for alt_acc in entry_to_uniprots[entry_name]:
                        if alt_acc in alias_to_string:
                            string_id = alias_to_string[alt_acc]
                            break

        # Strategy 2: CAFA3 -> Gene ID (e.g., FBgn) -> STRING (via aliases)
        if string_id is None and cafa3_id in cafa_to_id:
            gene_id = cafa_to_id[cafa3_id]  # e.g., FBgn0030647
            if gene_id in alias_to_string:
                string_id = alias_to_string[gene_id]
                mapped_cafa3_via_geneid += 1

        if string_id:
            if string_id not in needed_string_ids:
                needed_string_ids[string_id] = set()
            needed_string_ids[string_id].add(cafa3_id)
            mapped_cafa3 += 1

    logger.info(f"UniProt -> STRING: {mapped_uniprot}/{len(uniprot_ids)} ({mapped_uniprot/len(uniprot_ids)*100:.1f}%)")
    logger.info(f"  (via entry name: {mapped_uniprot_via_entry})")
    logger.info(f"CAFA3 -> STRING: {mapped_cafa3}/{len(cafa3_ids)} ({mapped_cafa3/len(cafa3_ids)*100:.1f}%)")
    logger.info(f"  (via gene ID: {mapped_cafa3_via_geneid})")
    logger.info(f"Total mapped: {mapped_uniprot + mapped_cafa3}/{len(all_protein_ids)}")
    logger.info(f"Need {len(needed_string_ids)} unique STRING embeddings")

    return needed_string_ids


def extract_and_save_ppi(needed_string_ids: dict, h5_file: Path,
                         output_dir: Path) -> int:
    """Extract PPI embeddings from h5 file and save by original protein ID."""
    logger.info("Extracting PPI embeddings from STRING h5 file...")

    output_dir.mkdir(parents=True, exist_ok=True)

    found_count = 0

    with h5py.File(h5_file, 'r') as f:
        for species_id in tqdm(f['species'].keys(), desc="Processing species"):
            proteins = f['species'][species_id]['proteins'][:]
            proteins = [p.decode('utf-8') for p in proteins]

            # Find needed proteins from this species
            needed_indices = []
            needed_mappings = []

            for idx, string_id in enumerate(proteins):
                if string_id in needed_string_ids:
                    needed_indices.append(idx)
                    needed_mappings.append((idx, needed_string_ids[string_id]))

            # Load and save
            if needed_indices:
                embeddings = f['species'][species_id]['embeddings'][:]

                for idx, protein_ids in needed_mappings:
                    emb = embeddings[idx]

                    for protein_id in protein_ids:
                        np.save(output_dir / f"{protein_id}.npy", emb)
                        found_count += 1

                del embeddings

    logger.info(f"Saved {found_count} PPI embeddings")
    return found_count


def print_coverage_stats(data_dir: Path, output_dir: Path):
    """Print coverage statistics."""
    logger.info("\nCoverage by Aspect and Split:")

    total_proteins = 0
    total_found = 0

    for aspect in ['BPO', 'CCO', 'MFO']:
        print(f"\n{aspect}:")

        for split in ['train', 'valid', 'test']:
            names_file = data_dir / f"{aspect}_{split}_names.npy"

            if not names_file.exists():
                continue

            proteins = np.load(names_file, allow_pickle=True)
            n_proteins = len(proteins)
            found = sum(1 for p in proteins if (output_dir / f"{p}.npy").exists())

            total_proteins += n_proteins
            total_found += found

            coverage = found / n_proteins * 100 if n_proteins > 0 else 0
            print(f"  {split:6s}: {n_proteins:5d} proteins | "
                  f"found: {found:5d} ({coverage:5.1f}%)")

    if total_proteins > 0:
        print(f"\nOverall: {total_found}/{total_proteins} "
              f"({total_found/total_proteins*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Extract PPI embeddings from STRING")
    parser.add_argument('--string-h5', type=str, default=STRING_H5_FILE,
                        help="Path to STRING embeddings h5 file")
    parser.add_argument('--string-alias', type=str, default=STRING_ALIAS_FILE,
                        help="Path to STRING alias file")
    parser.add_argument('--cafa-assessment-dir', type=str, default=CAFA_ASSESSMENT_DIR,
                        help="Path to CAFA assessment tool directory")
    parser.add_argument('--cafa3-id-mapping', type=str, default=CAFA3_ID_MAPPING,
                        help="Path to CAFA3 ID mapping JSON file (CAFA3 ID -> gene ID)")
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help="Data directory with protein names")
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help="Output directory for embeddings")
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit number of proteins (for testing)")

    args = parser.parse_args()

    h5_file = Path(args.string_h5)
    alias_file = Path(args.string_alias)
    cafa_assessment_dir = Path(args.cafa_assessment_dir)
    cafa3_id_mapping_file = Path(args.cafa3_id_mapping)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Check required files
    if not h5_file.exists():
        logger.error(f"STRING h5 file not found: {h5_file}")
        return
    if not alias_file.exists():
        logger.error(f"STRING alias file not found: {alias_file}")
        return
    if not cafa_assessment_dir.exists():
        logger.error(f"CAFA assessment dir not found: {cafa_assessment_dir}")
        return

    logger.info("=" * 70)
    logger.info("PPI EMBEDDING EXTRACTION")
    logger.info("=" * 70)

    # Step 1: Build CAFA -> UniProt mapping (for test set via CAFA assessment tool)
    cafa_to_uniprot = build_cafa_to_uniprot_mapping(cafa_assessment_dir)

    # Step 2: Build UniProt alternative mapping via entry names (for train/valid)
    uniprot_to_entry, entry_to_uniprots = build_uniprot_alternative_mapping(cafa_assessment_dir)

    # Step 3: Load CAFA3 ID -> gene ID mapping (for FlyBase/other gene IDs)
    cafa_to_id = {}
    if cafa3_id_mapping_file.exists():
        cafa_to_id = load_cafa3_id_mapping(cafa3_id_mapping_file)
    else:
        logger.warning(f"CAFA3 ID mapping not found: {cafa3_id_mapping_file}")

    # Step 4: Build ID -> STRING mapping (includes UniProt and FlyBase)
    alias_to_string = build_alias_to_string_mapping(alias_file)

    # Step 5: Find needed STRING IDs
    needed_string_ids = get_needed_string_ids(
        data_dir, cafa_to_uniprot, cafa_to_id, alias_to_string,
        uniprot_to_entry, entry_to_uniprots, args.limit
    )

    # Step 6: Extract and save
    extract_and_save_ppi(needed_string_ids, h5_file, output_dir)

    # Step 7: Print coverage
    print_coverage_stats(data_dir, output_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
