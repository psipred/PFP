"""
Extract UniProt text descriptions for CAFA proteins for GO function prediction.

Uses the same CAFA->UniProt mapping as PPI extraction, then queries UniProt REST API
to get: protein name, gene names, function, catalytic activity, pathway, 
similarity (family), subunit (complex), and subcellular location.

This version:
- Uses Multi-threading (ThreadPoolExecutor) to significantly speed up API queries.
- Reuses requests.Session objects per thread.
- Thread-safe file writing and caching.
- Cleans UniProt text (removes Rhea/CHEBI/GO IDs, EC numbers, extra spaces) for PubMedBERT.
"""

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


sys.path.append("/home/zijianzhou/project/PFP")


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

    # 1. Protein name
    if text_data.get('protein_name'):
        parts.append(text_data['protein_name'])

    # 2. Function
    if text_data.get('function'):
        parts.append(f"Function: {text_data['function']}")

    # 3. Catalytic Activity (Specific for MFO)
    if text_data.get('catalytic_activity'):
        parts.append(f"Catalytic activity: {text_data['catalytic_activity']}")

    # 4. Pathway (Specific for BPO)
    if text_data.get('pathway'):
        parts.append(f"Pathway: {text_data['pathway']}")
    
    # 5. Similarity (Family/Domain - High Priority)
    if text_data.get('similarity'):
        parts.append(f"Similarity: {text_data['similarity']}")

    # 6. Subcellular location (Specific for CCO)
    if text_data.get('subcellular_location'):
        parts.append(f"Location: {text_data['subcellular_location']}")
    
    # 7. Subunit (Complex info)
    if text_data.get('subunit'):
        parts.append(f"Subunit: {text_data['subunit']}")

    # 8. Gene names (ID-ish, lower priority if length limited)
    if text_data.get('gene_names'):
        parts.append(f"Gene: {text_data['gene_names']}")

    current_text = '. '.join([p for p in parts if p])

    # Additional cleaning & truncation
    current_text = clean_uniprot_text(current_text)

    if len(current_text) > max_length:
        current_text = current_text[:max_length]
        # Try to end at a sentence boundary
        last_period = current_text.rfind('.')
        if last_period > max_length * 0.7:
            current_text = current_text[:last_period + 1]

    return current_text.strip()


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


def main():
    """Main execution."""
    # Script is in: /home/zijianzhou/project/PFP/experiments/PLMs/utils/
    # Data is in:   /home/zijianzhou/project/PFP/experiments/PLMs/data/
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    cafa_assessment_dir = Path("/home/zijianzhou/project/CAFA_assessment_tool")
    output_file = data_dir / "embedding_cache" / "uniprot_text" / "protein_descriptions.tsv"
    checkpoint_file = output_file.parent / "processed_checkpoint.txt"

    print("=" * 70)
    print("UNIPROT TEXT EXTRACTION FOR GO FUNCTION PREDICTION")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_file}")

    # Step 1: Build CAFA -> UniProt mapping
    cafa_to_uniprot = build_cafa_to_uniprot_mapping(cafa_assessment_dir, data_dir)

    # Step 2: Extract and save text descriptions
    success, fail = extract_and_save_text(data_dir, cafa_to_uniprot, output_file, checkpoint_file)

    # Step 3: Print coverage stats
    print_coverage_stats(data_dir, output_file)

    # Step 4: Show samples
    show_sample_descriptions(output_file)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()