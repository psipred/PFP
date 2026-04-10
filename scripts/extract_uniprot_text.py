"""Build the UniProt text inputs kept by MMFP.

Kept paths:
- current UniProt text
- historical UniSave text with 2016-02-17 cutoff
- the canonical temporal mixed bundle
"""

import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np
import requests
import time
from tqdm import tqdm
from collections import defaultdict
import json
import re
import concurrent.futures
import threading
from datetime import datetime


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CAFA_ASSESSMENT_DIR = Path(os.environ.get("CAFA_ASSESSMENT_DIR", "CAFA_assessment_tool"))
UNIPROT_TEXT_DIR = DATA_DIR / "embedding_cache" / "uniprot_text"
CURRENT_OUTPUT_FILE = UNIPROT_TEXT_DIR / "protein_descriptions.tsv"
CURRENT_CHECKPOINT_FILE = UNIPROT_TEXT_DIR / "processed_checkpoint.txt"
HISTORICAL_OUTPUT_FILE = UNIPROT_TEXT_DIR / "protein_descriptions_historical.tsv"
HISTORICAL_CHECKPOINT_FILE = UNIPROT_TEXT_DIR / "historical_checkpoint.txt"
HISTORICAL_RAW_DIR = UNIPROT_TEXT_DIR / "historical_raw"
TEMPORAL_DIR = UNIPROT_TEXT_DIR / "temporal_recipe"
TEMPORAL_PUNCT_FILE = TEMPORAL_DIR / "protein_descriptions_historical_punct_v1_test.tsv"
MIXED_OUTPUT_FILE = TEMPORAL_DIR / "protein_descriptions_mixed.tsv"
TEXT_BUNDLE_METADATA = TEMPORAL_DIR / "metadata.json"
EXTERNAL_CURRENT_TEXT_FILE = Path(
    os.environ.get("EXTERNAL_TEXT_TSV", str(CURRENT_OUTPUT_FILE))
)
CUTOFF_DATE = "2016-02-17"
# ==============================
    # Helpers for CAFA ↔ UniProt mapping
# ==============================

def read_target_mapping(taxon, target_folder):
    """Read CAFA ID -> UniProt entry name mapping.

    Based on ID_conversion.py __read_target_mapping__ function.
    Returns: dict mapping CAFA ID to UniProt entry name
    """
    reduced_taxons = [
        '85962', '83333', '10116', '7955', '273057', '160488',
        '170187', '224308', '243273', '9606', '243232', '44689',
        '559292', '284812', '3702', '8355', '10090', '223283',
        '99287', '321314'
    ]

    targetdict = {}
    if taxon in reduced_taxons:
        if taxon in ['208963', '7227', '237561']:
            filename = f'mapping.{taxon}.map'
        else:
            filename = f'sp_species.{taxon}.map'

        filepath = Path(target_folder) / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found for taxon {taxon}")
            return targetdict

        with open(filepath, 'r') as handle:
            for line in handle:
                fields = line.strip().split('\t')
                if len(fields) >= 2:
                    cafaid = fields[0]
                    entry_name = fields[1]
                    targetdict[cafaid] = entry_name
    else:
        print(f'{taxon} is not a CAFA3 species')

    return targetdict


def read_uniprot_mapping(taxon, uniprot_folder):
    """Read UniProt entry name -> UniProt accession mapping.

    Based on ID_conversion.py __uniprot_mapping__ function.
    Returns: dict mapping UniProt entry name to accession
    """
    reduced_taxons = [
        '85962', '83333', '10116', '7955', '273057', '160488',
        '170187', '224308', '243273', '9606', '243232', '44689',
        '559292', '284812', '3702', '8355', '10090', '223283',
        '99287', '321314'
    ]

    uniprotdict = {}
    mapping = False

    if taxon not in reduced_taxons:
        print(f'{taxon} is not a CAFA3 species')
        return uniprotdict

    if taxon in [
        '10090', '10116', '284812', '3702', '44689', '559292',
        '7227', '7955', '83333', '9606'
    ]:
        filename = f'uniprot_ac_to_id_{taxon}.map'
        mapping = True
    else:
        filename = f'uniprot_ac_to_id_{taxon}.tab'

    filepath = Path(uniprot_folder) / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found for taxon {taxon}")
        return uniprotdict

    with open(filepath, 'r') as f:
        # Skip header line (present in both .map and .tab variants in CAFA tools)
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            # Expect at least accession + entry_name
            if len(parts) < 2:
                continue

            accession = parts[0]
            if mapping:
                # .map files: accession, ?, entry_name
                if len(parts) >= 3:
                    entry_name = parts[2]
                else:
                    # Fallback: skip malformed line
                    continue
            else:
                # .tab files: accession, entry_name
                entry_name = parts[1]

            if entry_name not in uniprotdict:
                uniprotdict[entry_name] = accession
            else:
                # Should be rare, but keep the first mapping
                print(f"Warning: Repeated UniProt entry name {entry_name}")

    return uniprotdict


def is_entry_name(protein_id):
    """Check if string looks like a UniProt entry name (e.g., RPAC2_DANRE).

    Entry names have format: NAME_ORGANISM (e.g., RPAC2_DANRE, INS_HUMAN)
    Accessions are alphanumeric IDs (e.g., P12345, Q7ZVF0)
    """
    if '_' in protein_id and protein_id.isupper():
        parts = protein_id.split('_')
        if len(parts) == 2 and all(part.isalnum() for part in parts):
            return True
    return False


def build_cafa_to_uniprot_mapping(cafa_assessment_dir, data_dir):
    """Build comprehensive CAFA ID -> UniProt Accession mapping.

    Uses the existing CAFA mapping infrastructure:
    CAFA ID -> UniProt Entry Name (from sp_species.*.map)
    UniProt Entry Name -> Accession (from uniprot_ac_to_id_*.map/tab)

    Args:
        cafa_assessment_dir: Path to CAFA_assessment_tool directory
        data_dir: Path to your data directory

    Returns:
        dict: CAFA ID -> (entry_name, accession, taxon)
    """
    print("\n=== Building CAFA -> UniProt Accession Mapping ===")

    # All CAFA3 taxons
    all_taxons = [
        '85962', '83333', '10116', '7955', '273057', '160488',
        '170187', '224308', '243273', '9606', '243232', '44689',
        '559292', '284812', '3702', '8355', '10090', '223283',
        '99287', '321314'
    ]

    target_folder = Path(cafa_assessment_dir) / "ID_conversion" / "CAFA_mapping"
    uniprot_folder = Path(cafa_assessment_dir) / "ID_conversion" / "uniprot_mapping"

    cafa_to_uniprot = {}  # cafa_id -> (entry_name, accession, taxon)
    stats = defaultdict(lambda: defaultdict(int))

    # Process each taxon
    for taxon in all_taxons:
        target_dict = read_target_mapping(taxon, target_folder)
        uniprot_dict = read_uniprot_mapping(taxon, uniprot_folder)

        mapped = 0
        missing_entry = 0
        for cafaid, entry_name in target_dict.items():
            if entry_name in uniprot_dict:
                accession = uniprot_dict[entry_name]
                cafa_to_uniprot[cafaid] = (entry_name, accession, taxon)
                mapped += 1
            else:
                # Store entry name even if no accession mapping
                cafa_to_uniprot[cafaid] = (entry_name, None, taxon)
                missing_entry += 1

        stats[taxon]['mapped'] = mapped
        stats[taxon]['missing_accession'] = missing_entry

        if mapped > 0:
            print(f"  Taxon {taxon}: {mapped} mappings")

    # Also check for any proteins in data splits not covered by taxon mappings
    print("\nChecking for proteins not in taxon mappings...")
    all_proteins = set()
    for aspect in ['BPO', 'CCO', 'MFO']:
        for split in ['train', 'valid', 'test']:
            names_file = Path(data_dir) / f"{aspect}_{split}_names.npy"
            if names_file.exists():
                proteins = np.load(names_file, allow_pickle=True)
                all_proteins.update(proteins)

    unmapped = []
    for protein_id in all_proteins:
        if protein_id not in cafa_to_uniprot:
            unmapped.append(protein_id)

    if unmapped:
        print(f"⚠️  Found {len(unmapped)} proteins not in taxon mappings")
        print(f"    These will be treated as direct UniProt IDs")
        # Try to use them directly - might be accessions or entry names
        for protein_id in unmapped:
            if is_entry_name(protein_id):
                cafa_to_uniprot[protein_id] = (protein_id, None, 'unknown')
            else:
                # Assume it's already an accession
                cafa_to_uniprot[protein_id] = (protein_id, protein_id, 'unknown')

    total_mapped = sum(s['mapped'] for s in stats.values())
    total_missing = sum(s['missing_accession'] for s in stats.values())

    print(f"\n{'='*70}")
    print(f"Total CAFA IDs: {len(cafa_to_uniprot)}")
    print(f"  With accession: {total_mapped}")
    print(f"  Need API resolution: {total_missing + len(unmapped)}")

    return cafa_to_uniprot


def get_needed_cafa_ids(data_dir):
    """Get all CAFA IDs from data splits."""
    print("\n=== Finding Required CAFA IDs ===")

    all_cafa_ids = set()
    for aspect in ['BPO', 'CCO', 'MFO']:
        for split in ['train', 'valid', 'test']:
            names_file = data_dir / f"{aspect}_{split}_names.npy"
            if names_file.exists():
                proteins = np.load(names_file, allow_pickle=True)
                all_cafa_ids.update(proteins)

    print(f"Total unique CAFA IDs: {len(all_cafa_ids)}")
    return all_cafa_ids


# ==============================
# UniProt API & text cleaning
# ==============================

def map_entry_name_to_accession(entry_name, session, cache, cache_lock=None, max_retries=3):
    """Map UniProt entry name to accession via API (Thread-safe if lock provided)."""
    
    # Check cache safely
    if cache_lock:
        with cache_lock:
            if entry_name in cache:
                return cache[entry_name]
    elif entry_name in cache:
        return cache[entry_name]

    url = f"https://rest.uniprot.org/uniprotkb/{entry_name}"
    base_timeout = 15

    result = (None, "Unknown error")

    for attempt in range(max_retries):
        timeout = base_timeout * (attempt + 1)
        try:
            response = session.get(
                url,
                timeout=timeout,
                headers={'Accept': 'application/json'}
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    accession = data.get('primaryAccession')
                    if accession:
                        result = (accession, None)
                        break
                    else:
                        result = (None, "Missing primaryAccession in response")
                        break
                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    result = (None, f"JSON decode error: {str(e)[:50]}")
                    break

            elif response.status_code == 404:
                result = (None, "Not found in UniProt")
                break

            elif response.status_code == 400:
                result = (None, "Invalid entry name format")
                break

            elif response.status_code in [500, 502, 503, 504]:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                result = (None, f"UniProt API HTTP {response.status_code}")
                break

            else:
                result = (None, f"HTTP {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                break

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            result = (None, f"Timeout after {max_retries} attempts (max {timeout}s)")
            break

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            result = (None, str(e)[:100])
            break
    
    # Update cache safely
    if cache_lock:
        with cache_lock:
            cache[entry_name] = result
    else:
        cache[entry_name] = result

    return result


def clean_uniprot_text(text):
    """Clean UniProt free text for PubMedBERT."""
    if not text:
        return text

    # Remove Rhea, ChEBI, GO IDs
    text = re.sub(r'RHEA:\d+', ' ', text)
    text = re.sub(r'CHEBI:\d+', ' ', text)
    text = re.sub(r'GO:\d{7}', ' ', text)

    # Remove EC numbers like EC=3.2.1.4
    text = re.sub(r'EC=\d+(\.\d+)*', ' ', text)

    # Remove evidence bracket patterns like {ECO:0000269|PubMed:12345678}
    text = re.sub(r'\{ECO:[^}]+\}', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_historical_description_artifacts(text):
    """Normalize punctuation artifacts common in cached historical descriptions.

    This is intentionally conservative and only targets formatting issues that
    show up after field assembly in the historical text cache, such as repeated
    sentence punctuation and inconsistent separator spacing.
    """
    if not text:
        return text

    # Collapse repeated sentence punctuation produced by stitched fields:
    # ". .", "..", ". ..", etc. -> ". "
    text = re.sub(r'(?:\s*\.\s*){2,}', '. ', text)

    # Normalize separator spacing while preserving the original field content.
    text = re.sub(r'\s*;\s*', '; ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+\.', '.', text)

    # Fix split hyphenation observed in historical records, e.g. "5- hydroxy".
    text = re.sub(r'(?<=\w)-\s+(?=\w)', '-', text)

    # Final whitespace collapse only; do not re-run field extraction logic.
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_uniprot_response(data):
    """Parse UniProt API response to extract text fields."""
    result = {
        'protein_name': '',
        'gene_names': '',
        'function': '',
        'catalytic_activity': '',
        'pathway': '',
        'similarity': '',
        'subunit': '',
        'subcellular_location': ''
    }

    # Protein name
    if 'proteinDescription' in data:
        rec_name = data['proteinDescription'].get('recommendedName', {})
        if 'fullName' in rec_name:
            result['protein_name'] = rec_name['fullName'].get('value', '')

    # Gene names
    if 'genes' in data and len(data['genes']) > 0:
        gene_list = []
        for gene in data['genes']:
            if 'geneName' in gene:
                gene_list.append(gene['geneName'].get('value', ''))
        result['gene_names'] = ', '.join(gene_list) if gene_list else ''

    # Comments processing
    if 'comments' in data:
        for comment in data['comments']:
            c_type = comment.get('commentType')

            # Function
            if c_type == 'FUNCTION':
                texts = comment.get('texts', [])
                if texts and not result['function']:
                    raw = texts[0].get('value', '')
                    result['function'] = clean_uniprot_text(raw)

            # Catalytic Activity
            elif c_type == 'CATALYTIC ACTIVITY':
                reaction = comment.get('reaction', {})
                if 'name' in reaction and not result['catalytic_activity']:
                    raw = reaction['name']
                    result['catalytic_activity'] = clean_uniprot_text(raw)

            # Pathway
            elif c_type == 'PATHWAY':
                texts = comment.get('texts', [])
                if texts and not result['pathway']:
                    raw = texts[0].get('value', '')
                    result['pathway'] = clean_uniprot_text(raw)

            # Similarity (Family/Domain info) - Very predictive for GO
            elif c_type == 'SIMILARITY':
                texts = comment.get('texts', [])
                if texts and not result['similarity']:
                    raw = texts[0].get('value', '')
                    result['similarity'] = clean_uniprot_text(raw)
            
            # Subunit (Complex info) - Predictive for CCO/BPO
            elif c_type == 'SUBUNIT':
                texts = comment.get('texts', [])
                if texts and not result['subunit']:
                    raw = texts[0].get('value', '')
                    result['subunit'] = clean_uniprot_text(raw)

            # Subcellular location
            elif c_type == 'SUBCELLULAR LOCATION':
                locations = []
                for loc in comment.get('subcellularLocations', []):
                    if 'location' in loc:
                        locations.append(loc['location'].get('value', ''))
                if locations and not result['subcellular_location']:
                    joined = ', '.join(locations)
                    result['subcellular_location'] = clean_uniprot_text(joined)

    # Clean protein_name and gene_names lightly (just whitespace)
    result['protein_name'] = clean_uniprot_text(result['protein_name'])
    result['gene_names'] = clean_uniprot_text(result['gene_names'])

    return result


def query_uniprot_text(mapping_info, entry_name_cache, session, cache_lock=None, max_retries=3):
    """Query UniProt REST API for text description fields (Thread-safe)."""
    entry_name, accession, taxon = mapping_info

    # Case 1: We have accession from local mapping - use it directly
    if accession:
        uniprot_id = accession

    # Case 2: We have entry name but no accession - resolve via API
    elif entry_name:
        # Check/Update cache with locking
        accession, map_error = map_entry_name_to_accession(
            entry_name, session, entry_name_cache, cache_lock=cache_lock
        )
        
        if map_error or not accession:
            return None
        uniprot_id = accession

    else:
        # No mapping at all
        return None

    # Clean up UniProt ID - remove isoform suffix if present
    clean_id = uniprot_id.split('-')[0].strip()

    base_url = "https://rest.uniprot.org/uniprotkb"

    # Request specific fields to minimize response size
    fields = [
        "protein_name",
        "gene_names",
        "cc_function",
        "cc_catalytic_activity",
        "cc_pathway",
        "cc_similarity",  # Added for GO prediction
        "cc_subunit",     # Added for GO prediction
        "cc_subcellular_location"
    ]

    url = f"{base_url}/{clean_id}"
    params = {"fields": ",".join(fields)}

    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return parse_uniprot_response(data)

            elif response.status_code == 404:
                return None  # Protein not found

            elif response.status_code == 429:  # Rate limit
                # Exponential backoff
                time.sleep(2 ** attempt)
                continue

            else:
                return None

        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None

    return None


# ==============================
# Text building & extraction
# ==============================

def build_text_description(text_data, max_length=500):
    """Build a concise text description for PubMedBERT input.

    Priority for GO Prediction:
    1. Protein name (Identity)
    2. Function (General context)
    3. Catalytic Activity (Strong MFO signal)
    4. Pathway (Strong BPO signal)
    5. Similarity (Family info - very strong prior)
    6. Subcellular location (Strong CCO signal)
    7. Subunit (Complex info - good for CCO/BPO)
    8. Gene names (Identity/Literature ref)
    """
    parts = []

    if text_data.get('protein_name'):
        parts.append(text_data['protein_name'])
    if text_data.get('function'):
        parts.append(f"Function: {text_data['function']}")
    if text_data.get('catalytic_activity'):
        parts.append(f"Catalytic activity: {text_data['catalytic_activity']}")
    if text_data.get('pathway'):
        parts.append(f"Pathway: {text_data['pathway']}")
    if text_data.get('similarity'):
        parts.append(f"Similarity: {text_data['similarity']}")
    if text_data.get('subcellular_location'):
        parts.append(f"Location: {text_data['subcellular_location']}")
    if text_data.get('subunit'):
        parts.append(f"Subunit: {text_data['subunit']}")
    if text_data.get('gene_names'):
        parts.append(f"Gene: {text_data['gene_names']}")

    text = clean_uniprot_text('. '.join(parts))
    if len(text) > max_length:
        text = text[:max_length]
        last_period = text.rfind('.')
        if last_period > max_length * 0.7:
            text = text[:last_period + 1]
    return text.strip()


def extract_and_save_text(data_dir, cafa_to_uniprot, output_file, checkpoint_file):
    """Extract text descriptions for all CAFA proteins and save to single file using threads."""
    print("\n=== Extracting UniProt Text Descriptions (Multi-threaded) ===")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_file = Path(checkpoint_file)
    processed = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            processed = set(line.strip() for line in f if line.strip())
        print(f"Loaded checkpoint: {len(processed)} already processed")

    needed_cafa_ids = get_needed_cafa_ids(data_dir)

    # Deterministic ordering
    to_process = sorted([cid for cid in needed_cafa_ids if cid not in processed])
    print(f"Need to process: {len(to_process)} CAFA IDs")
    
    if not to_process:
        return 0, 0

    with_local_accession = sum(
        1 for cid in to_process
        if cid in cafa_to_uniprot and cafa_to_uniprot[cid][1] is not None
    )
    need_api = sum(
        1 for cid in to_process
        if cid in cafa_to_uniprot and cafa_to_uniprot[cid][0] and not cafa_to_uniprot[cid][1]
    )
    no_mapping = sum(1 for cid in to_process if cid not in cafa_to_uniprot)

    print(f"\nMapping statistics:")
    print(f"  With local accession: {with_local_accession}")
    print(f"  Need API resolution: {need_api}")
    print(f"  No mapping: {no_mapping}")

    success_count = 0
    fail_count = 0
    
    # Shared resources
    entry_name_cache = {}
    cache_lock = threading.Lock()
    file_lock = threading.Lock()
    
    # Thread-local storage for Sessions
    thread_local = threading.local()

    def get_session():
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
        return thread_local.session

    def process_cafa_id(cafa_id):
        """Worker function for processing a single ID."""
        session = get_session()
        
        # Map to UniProt
        if cafa_id not in cafa_to_uniprot:
            return cafa_id, None, False

        mapping_info = cafa_to_uniprot[cafa_id]
        
        # Query UniProt (uses shared cache with lock)
        text_data = query_uniprot_text(mapping_info, entry_name_cache, session, cache_lock=cache_lock)

        if text_data is None:
            return cafa_id, None, False

        description = build_text_description(text_data)
        
        if description:
            return cafa_id, description, True
        return cafa_id, None, False

    # Multi-threading execution
    max_workers = 5  # Conservative limit for UniProt (avoid 429)
    
    with open(output_file, 'a') as outfile, open(checkpoint_file, 'a') as checkpoint:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_id = {executor.submit(process_cafa_id, cid): cid for cid in to_process}
            
            # Process as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(to_process), desc="Querying"):
                cafa_id = future_to_id[future]
                try:
                    cid, desc, success = future.result()
                    
                    # Thread-safe write (though we are in main thread here, it's sequential now)
                    if success and desc:
                        clean_desc = desc.replace('\n', ' ').replace('\t', ' ')
                        outfile.write(f"{cid}\t{clean_desc}\n")
                        outfile.flush()
                        success_count += 1
                    else:
                        fail_count += 1
                    
                    checkpoint.write(f"{cid}\n")
                    checkpoint.flush()
                    
                except Exception as e:
                    print(f"Error processing {cafa_id}: {e}")
                    fail_count += 1
                    checkpoint.write(f"{cafa_id}\n")

    print(f"\n=== Extraction Complete ===")
    print(f"Successfully extracted: {success_count}")
    print(f"Failed/Not found: {fail_count}")

    return success_count, fail_count


# ==============================
# Coverage + inspection
# ==============================

def print_coverage_stats(data_dir, output_file):
    """Print coverage statistics by aspect and split."""
    print("\n=== Coverage by Aspect and Split ===")

    output_file = Path(output_file)
    proteins_with_text = set()

    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    proteins_with_text.add(parts[0])

    print(f"Loaded {len(proteins_with_text)} proteins with text descriptions")

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
            found = sum(1 for p in proteins if p in proteins_with_text)

            total_proteins += n_proteins
            total_found += found

            coverage = found / n_proteins * 100 if n_proteins > 0 else 0
            print(
                f"  {split:6s}: {n_proteins:5d} proteins | "
                f"found: {found:5d} ({coverage:5.1f}%)"
            )

    print(f"\n=== Overall Summary ===")
    print(f"Total CAFA proteins: {total_proteins}")
    print(f"Text descriptions saved: {total_found}")
    if total_proteins > 0:
        print(f"Overall coverage: {total_found / total_proteins * 100:.2f}%")
    else:
        print(f"Overall coverage: N/A (no proteins found in data splits)")


def show_sample_descriptions(output_file, n_samples=5):
    """Show sample descriptions for inspection."""
    print(f"\n=== Sample Descriptions (first {n_samples}) ===")

    output_file = Path(output_file)
    if not output_file.exists():
        print("Output file not found")
        return

    count = 0
    with open(output_file, 'r') as f:
        for line in f:
            if count >= n_samples:
                break

            parts = line.strip().split('\t', 1)
            if len(parts) >= 2:
                cafa_id = parts[0]
                description = parts[1]

                print(f"\n{cafa_id}:")
                print(f"  Length: {len(description)} chars")
                preview = description[:200]
                print(f"  Text: {preview}{'...' if len(description) > 200 else ''}")

                count += 1


# ==============================
# Public helper & main
# ==============================

def load_protein_descriptions(tsv_file):
    """Helper function to load protein descriptions into a dictionary.

    Args:
        tsv_file: Path to the protein_descriptions.tsv file

    Returns:
        dict: {cafa_id: description}
    """
    descriptions = {}
    with open(tsv_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) >= 2:
                descriptions[parts[0]] = parts[1]
    return descriptions


def save_protein_descriptions(tsv_file, rows):
    """Save protein descriptions to TSV."""
    tsv_file = Path(tsv_file)
    tsv_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_file, 'w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        for protein_id, text in rows:
            writer.writerow([protein_id, text])


def get_split_protein_ids(data_dir=DATA_DIR, splits=None):
    """Get union of protein IDs for the requested splits across BPO/CCO/MFO."""
    if splits is None:
        splits = ['train', 'valid', 'test']
    ids = set()
    for aspect in ['BPO', 'CCO', 'MFO']:
        for split in splits:
            names_file = Path(data_dir) / f"{aspect}_{split}_names.npy"
            if names_file.exists():
                proteins = np.load(names_file, allow_pickle=True)
                ids.update(str(p) for p in proteins)
    return sorted(ids)


def materialize_current_text_source(output_file=CURRENT_OUTPUT_FILE, source_file=EXTERNAL_CURRENT_TEXT_FILE):
    """Copy the original extracted current-text TSV into the MMFP data tree if needed."""
    output_file = Path(output_file)
    if output_file.exists():
        return output_file
    source_file = Path(source_file)
    if not source_file.exists():
        raise FileNotFoundError(
            f"Current text TSV not found at {source_file}. "
            "Run current extraction first or restore the original PLMs text TSV."
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(source_file.read_text())
    return output_file


def parse_unisave_date(date_str):
    """Parse UniSave date format 'DD-Mon-YYYY'."""
    try:
        return datetime.strptime(date_str, "%d-%b-%Y")
    except (ValueError, TypeError):
        return None


def get_unisave_versions(accession, session, max_retries=3):
    """Get all versions of a UniProt entry from UniSave."""
    url = f"https://rest.uniprot.org/unisave/{accession}?format=json"
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('results', data) if isinstance(data, dict) else data
                return results if isinstance(results, list) else []
            if resp.status_code == 404:
                return []
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return []
        except (requests.RequestException, json.JSONDecodeError):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return []


def find_historical_version(versions, cutoff_date=CUTOFF_DATE):
    """Find the latest UniSave version with firstReleaseDate <= cutoff_date."""
    cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
    valid = []
    for version in versions:
        release_date = version.get('firstReleaseDate', '')
        if not release_date:
            continue
        release_dt = parse_unisave_date(release_date)
        if release_dt and release_dt <= cutoff_dt:
            valid.append(version)
    if not valid:
        return None
    valid.sort(key=lambda item: item.get('entryVersion', 0))
    return valid[-1].get('entryVersion')


def get_historical_record(accession, version, session, max_retries=3):
    """Fetch a specific historical UniProt entry version as flat text."""
    url = f"https://rest.uniprot.org/unisave/{accession}?format=txt&versions={version}"
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def parse_uniprot_flattext(text):
    """Parse UniProt flat-text into the same fields as parse_uniprot_response()."""
    result = {
        'protein_name': '',
        'gene_names': '',
        'function': '',
        'catalytic_activity': '',
        'pathway': '',
        'similarity': '',
        'subunit': '',
        'subcellular_location': '',
    }
    if not text:
        return result

    lines = text.split('\n')

    de_lines = [line[5:] for line in lines if line.startswith('DE   ')]
    for de_line in de_lines:
        match = re.search(r'RecName:\s*Full=([^;{]+)', de_line)
        if match and not result['protein_name']:
            result['protein_name'] = match.group(1).strip()
        if not result['protein_name']:
            match = re.search(r'SubName:\s*Full=([^;{]+)', de_line)
            if match:
                result['protein_name'] = match.group(1).strip()

    gn_lines = [line[5:] for line in lines if line.startswith('GN   ')]
    gene_names = []
    for gn_line in gn_lines:
        match = re.search(r'Name=([^;{]+)', gn_line)
        if match:
            gene_names.append(match.group(1).strip())
    result['gene_names'] = ', '.join(gene_names)

    cc_text = []
    skip_block = False
    for line in lines:
        if line.startswith('CC   -----'):
            skip_block = not skip_block
            continue
        if skip_block:
            continue
        if line.startswith('CC   '):
            cc_text.append(line[5:])
    full_cc = '\n'.join(cc_text)

    comment_types = {
        'FUNCTION': 'function',
        'CATALYTIC ACTIVITY': 'catalytic_activity',
        'PATHWAY': 'pathway',
        'SIMILARITY': 'similarity',
        'SUBUNIT': 'subunit',
        'SUBCELLULAR LOCATION': 'subcellular_location',
    }
    for cc_type, field_name in comment_types.items():
        pattern = rf'-!-\s*{re.escape(cc_type)}:\s*(.*?)(?=-!-|$)'
        match = re.search(pattern, full_cc, re.DOTALL)
        if not match:
            continue
        raw_text = match.group(1).strip()
        raw_text = re.sub(r'\n\s*', ' ', raw_text)
        raw_text = re.sub(r'\s+', ' ', raw_text)
        if cc_type == 'CATALYTIC ACTIVITY':
            reaction_match = re.search(r'Reaction=([^;]+)', raw_text)
            if reaction_match:
                raw_text = reaction_match.group(1).strip()
        if cc_type == 'SUBCELLULAR LOCATION':
            raw_text = re.sub(r'Note=.*', '', raw_text).strip()
            raw_text = re.sub(r'\{[^}]+\}', '', raw_text)
            raw_text = raw_text.strip('. ')
        result[field_name] = clean_uniprot_text(raw_text)

    result['protein_name'] = clean_uniprot_text(result['protein_name'])
    result['gene_names'] = clean_uniprot_text(result['gene_names'])
    return result


def resolve_special_ids_via_uniprot(cafa_ids_with_info):
    """Resolve CAFA IDs from special taxons to UniProt accessions."""
    if not cafa_ids_with_info:
        return {}

    resolved = {}
    by_taxon = defaultdict(list)
    for cafa_id, entry_name, taxon in cafa_ids_with_info:
        by_taxon[taxon].append((cafa_id, entry_name))

    session = requests.Session()
    for taxon, entries in by_taxon.items():
        print(f"  Resolving {len(entries)} IDs for taxon {taxon}...")
        for cafa_id, entry_name in entries:
            url = f"https://rest.uniprot.org/uniprotkb/{entry_name}"
            try:
                resp = session.get(url, timeout=15, headers={'Accept': 'application/json'})
                if resp.status_code == 200:
                    data = resp.json()
                    accession = data.get('primaryAccession')
                    if accession:
                        resolved[cafa_id] = accession
                        continue
            except (requests.RequestException, json.JSONDecodeError):
                pass

            try:
                search_url = "https://rest.uniprot.org/uniprotkb/search"
                params = {
                    'query': f'(gene:{entry_name} OR id:{entry_name}) AND organism_id:{taxon}',
                    'format': 'json',
                    'size': 1,
                }
                resp = session.get(search_url, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get('results', [])
                    if results:
                        accession = results[0].get('primaryAccession')
                        if accession:
                            resolved[cafa_id] = accession
            except (requests.RequestException, json.JSONDecodeError):
                pass

            time.sleep(0.5)
    return resolved


def process_single_historical_protein(cafa_id, accession, session, raw_dir):
    """Fetch, parse, and build a historical text description for one protein."""
    if not accession:
        return cafa_id, None, "no_accession"

    raw_dir = Path(raw_dir)
    raw_file = raw_dir / f"{cafa_id}_{accession}.txt"
    raw_text = None

    if raw_file.exists():
        raw_text = raw_file.read_text()
    else:
        versions = get_unisave_versions(accession, session)
        if not versions:
            return cafa_id, None, "no_versions"

        version_num = find_historical_version(versions)
        if version_num is None:
            return cafa_id, None, "no_historical_version"

        raw_text = get_historical_record(accession, version_num, session)
        if not raw_text:
            return cafa_id, None, "fetch_failed"
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_file.write_text(raw_text)
        except OSError:
            pass

    text_data = parse_uniprot_flattext(raw_text)
    description = build_text_description(text_data)
    if description:
        return cafa_id, description, "ok"
    return cafa_id, None, "empty_description"


def extract_historical_text(
    data_dir=DATA_DIR,
    cafa_assessment_dir=CAFA_ASSESSMENT_DIR,
    output_file=HISTORICAL_OUTPUT_FILE,
    checkpoint_file=HISTORICAL_CHECKPOINT_FILE,
    raw_dir=HISTORICAL_RAW_DIR,
    splits=None,
    workers=5,
):
    """Extract historical text descriptions for the requested splits."""
    if splits is None:
        splits = ['test']

    output_file = Path(output_file)
    checkpoint_file = Path(checkpoint_file)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Historical UniProt Extraction ({', '.join(splits)}) ===")
    print(f"Cutoff date: {CUTOFF_DATE}")
    test_ids = get_split_protein_ids(data_dir, splits=splits)
    print(f"Proteins to cover: {len(test_ids)}")

    cafa_to_uniprot = build_cafa_to_uniprot_mapping(cafa_assessment_dir, data_dir)
    test_accessions = {}
    needs_api_resolution = []
    for cafa_id in test_ids:
        if cafa_id in cafa_to_uniprot:
            entry_name, accession, taxon = cafa_to_uniprot[cafa_id]
            if accession:
                test_accessions[cafa_id] = accession
            else:
                needs_api_resolution.append((cafa_id, entry_name, taxon))
        else:
            needs_api_resolution.append((cafa_id, cafa_id, 'unknown'))

    if needs_api_resolution:
        resolved = resolve_special_ids_via_uniprot(needs_api_resolution)
        test_accessions.update(resolved)
    print(f"Mapped accessions: {len(test_accessions)} / {len(test_ids)}")

    processed = set()
    if checkpoint_file.exists():
        processed = {line.strip() for line in checkpoint_file.read_text().splitlines() if line.strip()}
        print(f"Loaded checkpoint: {len(processed)} processed IDs")
    to_process = sorted(cid for cid in test_ids if cid not in processed)
    if not to_process:
        print("All requested proteins already processed.")
        return 0, 0, {"skipped": len(test_ids)}

    success_count = 0
    fail_count = 0
    status_counts = defaultdict(int)
    thread_local = threading.local()

    def get_session():
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
        return thread_local.session

    def worker(cafa_id):
        return process_single_historical_protein(
            cafa_id,
            test_accessions.get(cafa_id),
            get_session(),
            raw_dir,
        )

    with open(output_file, 'a') as outfile, open(checkpoint_file, 'a') as ckpt:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_id = {executor.submit(worker, cid): cid for cid in to_process}
            for future in tqdm(
                concurrent.futures.as_completed(future_to_id),
                total=len(to_process),
                desc="Fetching historical records",
            ):
                cafa_id = future_to_id[future]
                try:
                    cid, desc, status = future.result()
                    status_counts[status] += 1
                    if desc:
                        clean_desc = desc.replace('\n', ' ').replace('\t', ' ')
                        outfile.write(f"{cid}\t{clean_desc}\n")
                        outfile.flush()
                        success_count += 1
                    else:
                        fail_count += 1
                    ckpt.write(f"{cid}\n")
                    ckpt.flush()
                except Exception as exc:
                    print(f"Error processing {cafa_id}: {exc}")
                    fail_count += 1
                    ckpt.write(f"{cafa_id}\n")
                    ckpt.flush()

    return success_count, fail_count, dict(status_counts)


def build_historical_punct_v1_test_tsv(
    historical_tsv=HISTORICAL_OUTPUT_FILE,
    output_tsv=TEMPORAL_PUNCT_FILE,
    data_dir=DATA_DIR,
):
    """Build the accepted temporal text variant from historical test descriptions."""
    historical = load_protein_descriptions(historical_tsv)
    test_ids = get_split_protein_ids(data_dir, splits=['test'])
    rows = []
    changed = 0
    missing = []
    for protein_id in test_ids:
        baseline = historical.get(protein_id)
        if baseline is None:
            missing.append(protein_id)
            continue
        cleaned = clean_historical_description_artifacts(baseline)
        if cleaned != baseline:
            changed += 1
        rows.append((protein_id, cleaned))
    save_protein_descriptions(output_tsv, rows)
    return {
        'output_tsv': str(output_tsv),
        'records': len(rows),
        'changed_records': changed,
        'missing_records': len(missing),
    }


def build_mixed_temporal_tsv(
    current_tsv=CURRENT_OUTPUT_FILE,
    hist_test_tsv=TEMPORAL_PUNCT_FILE,
    output_tsv=MIXED_OUTPUT_FILE,
    bundle_dir=TEMPORAL_DIR,
    historical_tsv=HISTORICAL_OUTPUT_FILE,
    data_dir=DATA_DIR,
):
    """Build a mixed TSV: current text for train/valid, historical text for test."""
    current = load_protein_descriptions(current_tsv)
    historical_test = load_protein_descriptions(hist_test_tsv)

    train_valid_ids = get_split_protein_ids(data_dir, splits=['train', 'valid'])
    test_ids = get_split_protein_ids(data_dir, splits=['test'])
    overlap = set(train_valid_ids) & set(test_ids)
    if overlap:
        sample = ", ".join(sorted(list(overlap))[:5])
        raise ValueError(f"Train/valid and test IDs overlapped unexpectedly: {sample}")

    rows = []
    missing_current = []
    missing_hist = []
    for protein_id in train_valid_ids:
        text = current.get(protein_id)
        if text is None:
            missing_current.append(protein_id)
            continue
        rows.append((protein_id, text))
    for protein_id in test_ids:
        text = historical_test.get(protein_id)
        if text is None:
            missing_hist.append(protein_id)
            continue
        rows.append((protein_id, text))
    rows.sort(key=lambda item: item[0])
    save_protein_descriptions(output_tsv, rows)

    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    current_bundle = bundle_dir / "protein_descriptions_current.tsv"
    historical_bundle = bundle_dir / "protein_descriptions_historical.tsv"
    if not current_bundle.exists():
        current_bundle.write_text(Path(current_tsv).read_text())
    if not historical_bundle.exists():
        historical_bundle.write_text(Path(historical_tsv).read_text())

    metadata = {
        'current_tsv': str(current_tsv),
        'historical_tsv': str(historical_tsv),
        'historical_punct_test_tsv': str(hist_test_tsv),
        'mixed_tsv': str(output_tsv),
        'train_valid_records': len(train_valid_ids),
        'test_records': len(test_ids),
        'mixed_records': len(rows),
        'missing_current_records': len(missing_current),
        'missing_historical_test_records': len(missing_hist),
    }
    TEXT_BUNDLE_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with open(TEXT_BUNDLE_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata


def run_current_extraction(
    data_dir=DATA_DIR,
    cafa_assessment_dir=CAFA_ASSESSMENT_DIR,
    output_file=CURRENT_OUTPUT_FILE,
    checkpoint_file=CURRENT_CHECKPOINT_FILE,
):
    """Run current UniProt extraction."""
    print("=" * 70)
    print("CURRENT UNIPROT TEXT EXTRACTION FOR GO FUNCTION PREDICTION")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_file}")

    cafa_to_uniprot = build_cafa_to_uniprot_mapping(cafa_assessment_dir, data_dir)
    success, fail = extract_and_save_text(data_dir, cafa_to_uniprot, output_file, checkpoint_file)
    print_coverage_stats(data_dir, output_file)
    show_sample_descriptions(output_file)
    print(f"\n✓ Done! success={success}, fail={fail}")
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build current and temporal UniProt text inputs")
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('extract-current', help='Run current UniProt extraction into protein_descriptions.tsv')

    hist_parser = subparsers.add_parser('extract-historical', help='Run historical UniSave extraction')
    hist_parser.add_argument('--workers', type=int, default=5,
                             help='Number of historical extraction workers')
    subparsers.add_parser(
        'prepare-temporal-text',
        help='Materialize current source and build the canonical temporal text bundle',
    )
    return parser


def main(argv=None):
    """Main execution."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    command = args.command or 'extract-current'

    if command == 'extract-current':
        return run_current_extraction()

    if command == 'extract-historical':
        success, fail, status_counts = extract_historical_text(splits=['test'], workers=args.workers)
        print(f"✓ Historical extraction finished: success={success}, fail={fail}")
        print(json.dumps(status_counts, indent=2, sort_keys=True))
        return 0

    if command == 'prepare-temporal-text':
        materialize_current_text_source()
        build_historical_punct_v1_test_tsv()
        metadata = build_mixed_temporal_tsv()
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
