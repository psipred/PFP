"""Check AlphaFold structure availability for CAFA3 proteins via API.

Uses existing CAFA ID conversion infrastructure:
CAFA ID -> UniProt Entry Name -> UniProt Accession -> AlphaFold

Run this on your local machine or server with internet access.
Supports parallel processing with multiple workers for faster execution.
"""

from pathlib import Path
import numpy as np
import requests
from tqdm import tqdm
import time
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys
import os


def read_target_mapping(taxon, target_folder):
    """Read CAFA ID -> UniProt entry name mapping.
    
    Based on ID_conversion.py __read_target_mapping__ function.
    Returns: dict mapping CAFA ID to UniProt entry name
    """
    reduced_taxons = ['85962', '83333', '10116', '7955', '273057', '160488', 
                     '170187', '224308', '243273', '9606', '243232', '44689', 
                     '559292', '284812', '3702', '8355', '10090', '223283', 
                     '99287', '321314']
    
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
    reduced_taxons = ['85962', '83333', '10116', '7955', '273057', '160488', 
                     '170187', '224308', '243273', '9606', '243232', '44689', 
                     '559292', '284812', '3702', '8355', '10090', '223283', 
                     '99287', '321314']
    
    uniprotdict = {}
    mapping = False
    
    if taxon not in reduced_taxons:
        print(f'{taxon} is not a CAFA3 species')
        return uniprotdict
    
    if taxon in ['10090', '10116', '284812', '3702', '44689', '559292', 
                 '7227', '7955', '83333', '9606']:
        filename = f'uniprot_ac_to_id_{taxon}.map'
        mapping = True
    else:
        filename = f'uniprot_ac_to_id_{taxon}.tab'
    
    filepath = Path(uniprot_folder) / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found for taxon {taxon}")
        return uniprotdict
    
    with open(filepath, 'r') as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            accession = parts[0]
            entry_name = parts[2] if mapping else parts[1]
            
            if entry_name not in uniprotdict:
                uniprotdict[entry_name] = accession
            else:
                print(f"Warning: Repeated UniProt entry name {entry_name}")
    
    return uniprotdict


def build_cafa_to_accession_mapping(cafa_assessment_dir, data_dir):
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
    all_taxons = ['85962', '83333', '10116', '7955', '273057', '160488', 
                  '170187', '224308', '243273', '9606', '243232', '44689', 
                  '559292', '284812', '3702', '8355', '10090', '223283', 
                  '99287', '321314']
    
    target_folder = Path(cafa_assessment_dir) / "ID_conversion" / "CAFA_mapping"
    uniprot_folder = Path(cafa_assessment_dir) / "ID_conversion" / "uniprot_mapping"
    
    cafa_to_accession = {}
    stats = defaultdict(lambda: defaultdict(int))
    
    # Process each taxon
    for taxon in all_taxons:
        print(f"\nProcessing taxon {taxon}...")
        
        # Read mappings
        target_dict = read_target_mapping(taxon, target_folder)
        uniprot_dict = read_uniprot_mapping(taxon, uniprot_folder)
        
        # Combine mappings
        mapped = 0
        missing_entry = 0
        for cafaid, entry_name in target_dict.items():
            if entry_name in uniprot_dict:
                accession = uniprot_dict[entry_name]
                cafa_to_accession[cafaid] = (entry_name, accession, taxon)
                mapped += 1
            else:
                # Store entry name even if no accession mapping
                cafa_to_accession[cafaid] = (entry_name, None, taxon)
                missing_entry += 1
        
        stats[taxon]['mapped'] = mapped
        stats[taxon]['missing_accession'] = missing_entry
        
        print(f"  Mapped: {mapped}")
        if missing_entry > 0:
            print(f"  Missing accession: {missing_entry}")
    
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
        if protein_id not in cafa_to_accession:
            unmapped.append(protein_id)
    
    if unmapped:
        print(f"⚠️  Found {len(unmapped)} proteins not in taxon mappings")
        print(f"   These will be treated as direct UniProt IDs")
        # Try to use them directly - might be accessions or entry names
        for protein_id in unmapped:
            if is_entry_name(protein_id):
                cafa_to_accession[protein_id] = (protein_id, None, 'unknown')
            else:
                # Assume it's already an accession
                cafa_to_accession[protein_id] = (protein_id, protein_id, 'unknown')
    
    total_mapped = sum(s['mapped'] for s in stats.values())
    total_missing = sum(s['missing_accession'] for s in stats.values())
    
    print(f"\n{'='*70}")
    print(f"Total CAFA IDs: {len(cafa_to_accession)}")
    print(f"  With accession: {total_mapped}")
    print(f"  Need API resolution: {total_missing + len(unmapped)}")
    
    return cafa_to_accession


def get_all_cafa_proteins(data_dir):
    """Get all unique CAFA protein IDs from train/valid/test splits."""
    print("\n=== Collecting CAFA Protein IDs ===")
    
    all_proteins = set()
    stats = defaultdict(lambda: defaultdict(int))
    
    for aspect in ['BPO', 'CCO', 'MFO']:
        for split in ['train', 'valid', 'test']:
            names_file = Path(data_dir) / f"{aspect}_{split}_names.npy"
            
            if names_file.exists():
                proteins = np.load(names_file, allow_pickle=True)
                all_proteins.update(proteins)
                stats[aspect][split] = len(proteins)
    
    print("\nProteins per aspect and split:")
    for aspect in ['BPO', 'CCO', 'MFO']:
        print(f"\n{aspect}:")
        for split in ['train', 'valid', 'test']:
            if split in stats[aspect]:
                print(f"  {split:6s}: {stats[aspect][split]:5d} proteins")
    
    print(f"\nTotal unique CAFA proteins: {len(all_proteins)}")
    return all_proteins


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


def map_entry_name_to_accession(entry_name, session, cache={}):
    """Map UniProt entry name to accession via API.
    
    Only called for entry names without local mapping.
    Returns: (accession: str or None, error: str or None)
    """
    if entry_name in cache:
        return cache[entry_name]
    
    url = f"https://rest.uniprot.org/uniprotkb/{entry_name}"
    
    max_retries = 3
    base_timeout = 15
    
    for attempt in range(max_retries):
        timeout = base_timeout * (attempt + 1)
        
        try:
            response = session.get(url, timeout=timeout, headers={'Accept': 'application/json'})
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    accession = data.get('primaryAccession')
                    if accession:
                        cache[entry_name] = (accession, None)
                        return accession, None
                    else:
                        cache[entry_name] = (None, "Missing primaryAccession in response")
                        return None, "Missing primaryAccession in response"
                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        error = f"JSON decode error: {str(e)[:50]}"
                        cache[entry_name] = (None, error)
                        return None, error
            
            elif response.status_code == 404:
                cache[entry_name] = (None, "Not found in UniProt")
                return None, "Not found in UniProt"
            
            elif response.status_code == 400:
                cache[entry_name] = (None, "Invalid entry name format")
                return None, "Invalid entry name format"
            
            elif response.status_code in [500, 502, 503, 504]:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    error = f"UniProt API HTTP {response.status_code}"
                    cache[entry_name] = (None, error)
                    return None, error
            
            else:
                error = f"HTTP {response.status_code}"
                if attempt == max_retries - 1:
                    cache[entry_name] = (None, error)
                    return None, error
                time.sleep(1 * (attempt + 1))
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            else:
                error = f"Timeout after {max_retries} attempts (max {timeout}s)"
                cache[entry_name] = (None, error)
                return None, error
        
        except Exception as e:
            error_msg = str(e)[:100]
            if attempt == max_retries - 1:
                cache[entry_name] = (None, error_msg)
                return None, error_msg
            time.sleep(1 * (attempt + 1))
    
    error = "Unknown error after retries"
    cache[entry_name] = (None, error)
    return None, error


def check_alphafold_entry(cafa_id, mapping_info, entry_name_cache, cache_lock):
    """Check if AlphaFold has structure for given CAFA ID.
    
    mapping_info: tuple of (entry_name, accession, taxon)
    - If accession is available from local mapping, use it directly
    - If not, resolve entry_name via API
    
    Returns: (exists: bool, version: str or None, error: str or None, pdb_url: str or None, 
              resolved_accession: str or None)
    """
    session = requests.Session()
    
    entry_name, accession, taxon = mapping_info
    resolved_accession = None
    
    # Case 1: We have accession from local mapping - use it directly
    if accession:
        uniprot_id = accession
        resolved_accession = accession
    
    # Case 2: We have entry name but no accession - resolve via API
    elif entry_name:
        # Thread-safe cache access
        with cache_lock:
            if entry_name in entry_name_cache:
                cached = entry_name_cache[entry_name]
                accession, map_error = cached
            else:
                cached = None
        
        if cached is None:
            accession, map_error = map_entry_name_to_accession(entry_name, session)
            with cache_lock:
                entry_name_cache[entry_name] = (accession, map_error)
        
        if map_error:
            session.close()
            return False, None, f"Entry name mapping failed: {map_error}", None, None
        if not accession:
            session.close()
            return False, None, "Could not resolve entry name to accession", None, None
        
        resolved_accession = accession
        uniprot_id = accession
    
    else:
        # No mapping at all
        session.close()
        return False, None, "No UniProt mapping available", None, None
    
    # Clean up UniProt ID - remove isoform suffix if present
    clean_id = uniprot_id.split('-')[0].strip()
    
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{clean_id}"
    
    max_retries = 3
    base_timeout = 15
    
    for attempt in range(max_retries):
        timeout = base_timeout * (attempt + 1)
        
        try:
            response = session.get(url, timeout=timeout)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    entry = data[0]
                    version = entry.get('latestVersion', 'unknown')
                    pdb_url = entry.get('pdbUrl', None)
                    session.close()
                    return True, version, None, pdb_url, resolved_accession
                session.close()
                return False, None, "Empty response", None, resolved_accession
            
            elif response.status_code == 404:
                session.close()
                return False, None, None, None, resolved_accession
            
            elif response.status_code == 400:
                session.close()
                return False, None, f"Invalid ID format: {clean_id}", None, resolved_accession
            
            else:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                session.close()
                return False, None, f"HTTP {response.status_code}", None, resolved_accession
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            else:
                session.close()
                return False, None, f"Timeout after {max_retries} attempts", None, resolved_accession
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            else:
                session.close()
                return False, None, str(e)[:100], None, resolved_accession
    
    session.close()
    return False, None, "Unknown error after retries", None, resolved_accession


def process_single_protein(cafa_id, mapping_info, entry_name_cache, cache_lock):
    """Process a single protein - for parallel execution."""
    exists, version, error, pdb_url, resolved_accession = check_alphafold_entry(
        cafa_id, mapping_info, entry_name_cache, cache_lock
    )
    
    entry_name, accession, taxon = mapping_info
    
    if exists:
        return (cafa_id, 'found', (entry_name, accession, resolved_accession, version, pdb_url, taxon))
    elif error:
        return (cafa_id, 'error', (entry_name, accession, error, taxon))
    else:
        return (cafa_id, 'not_found', (entry_name, accession, resolved_accession, taxon))


def check_alphafold_coverage(cafa_proteins, cafa_to_accession, output_file=None, num_workers=10):
    """Check AlphaFold availability for all CAFA proteins using parallel workers."""
    print("\n=== Checking AlphaFold Coverage ===")
    print(f"AlphaFold API: https://alphafold.ebi.ac.uk/api/prediction/")
    print(f"Number of workers: {num_workers}")
    
    results = {
        'found': [],           # (cafa_id, entry_name, local_accession, resolved_accession, version, pdb_url, taxon)
        'not_found': [],       # (cafa_id, entry_name, local_accession, resolved_accession, taxon)
        'no_mapping': [],      # cafa_id
        'errors': []           # (cafa_id, entry_name, local_accession, error, taxon)
    }
    
    error_counts = defaultdict(int)
    entry_name_cache = {}
    cache_lock = Lock()
    
    api_resolutions = 0
    local_mappings = 0
    
    # Show mapping statistics
    print("\nMapping statistics:")
    with_local_accession = sum(1 for _, info in cafa_to_accession.items() if info[1] is not None)
    with_entry_only = sum(1 for _, info in cafa_to_accession.items() if info[0] and not info[1])
    print(f"  With local accession: {with_local_accession}")
    print(f"  Need API resolution: {with_entry_only}")
    
    # Filter proteins that have mappings
    proteins_to_check = []
    for cafa_id in sorted(cafa_proteins):
        if cafa_id not in cafa_to_accession:
            results['no_mapping'].append(cafa_id)
        else:
            proteins_to_check.append((cafa_id, cafa_to_accession[cafa_id]))
    
    print(f"\nProcessing {len(proteins_to_check)} proteins with {num_workers} workers...")
    
    # Process in parallel
    shown_found = 0
    shown_errors = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_protein, cafa_id, mapping_info, entry_name_cache, cache_lock): (cafa_id, mapping_info)
            for cafa_id, mapping_info in proteins_to_check
        }
        
        with tqdm(total=len(futures), desc="Checking AlphaFold") as pbar:
            for future in as_completed(futures):
                cafa_id, result_type, result_data = future.result()
                mapping_info = futures[future][1]
                entry_name, local_accession, taxon = mapping_info
                
                if result_type == 'found':
                    entry_name, local_acc, resolved_acc, version, pdb_url, taxon = result_data
                    results['found'].append((cafa_id, entry_name, local_acc, resolved_acc, version, pdb_url, taxon))
                    
                    if local_acc:
                        local_mappings += 1
                    elif resolved_acc:
                        api_resolutions += 1
                    
                    if shown_found < 5:
                        source = "local" if local_acc else f"API: {entry_name}→{resolved_acc}"
                        tqdm.write(f"  ✓ Found: {cafa_id} [{taxon}] ({source}) - v{version}")
                        shown_found += 1
                
                elif result_type == 'error':
                    entry_name, local_acc, error, taxon = result_data
                    results['errors'].append((cafa_id, entry_name, local_acc, error, taxon))
                    error_type = error.split(':')[0] if ':' in error else error.split(' ')[0]
                    error_counts[error_type] += 1
                    
                    if shown_errors < 5:
                        tqdm.write(f"  ✗ Error: {cafa_id} [{taxon}] ({entry_name}) - {error}")
                        shown_errors += 1
                
                else:  # not_found
                    entry_name, local_acc, resolved_acc, taxon = result_data
                    results['not_found'].append((cafa_id, entry_name, local_acc, resolved_acc, taxon))
                
                pbar.update(1)
    
    # Print summary
    print("\n" + "="*70)
    print("=== Coverage Summary ===")
    print("="*70)
    total = len(cafa_proteins)
    found = len(results['found'])
    not_found = len(results['not_found'])
    no_mapping = len(results['no_mapping'])
    errors = len(results['errors'])
    
    print(f"Total CAFA proteins: {total}")
    print(f"  ✓ Found in AlphaFold: {found} ({found/total*100:.1f}%)")
    print(f"  ✗ Not in AlphaFold: {not_found} ({not_found/total*100:.1f}%)")
    print(f"  ? No UniProt mapping: {no_mapping} ({no_mapping/total*100:.1f}%)")
    print(f"  ! API errors: {errors} ({errors/total*100:.1f}%)")
    
    print(f"\nMapping method breakdown:")
    print(f"  Local mapping used: {local_mappings} ({local_mappings/max(1,found)*100:.1f}% of found)")
    print(f"  API resolution used: {api_resolutions} ({api_resolutions/max(1,found)*100:.1f}% of found)")
    
    if error_counts:
        print("\nError breakdown:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {error_type}: {count}")
    
    mappable = total - no_mapping
    if mappable > 0:
        print(f"\nOf proteins with UniProt mapping ({mappable}):")
        print(f"  Coverage: {found/mappable*100:.1f}%")
    
    # Save detailed results
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("=== AlphaFold Coverage for CAFA3 Proteins ===\n\n")
            
            f.write(f"Total proteins: {total}\n")
            f.write(f"Found in AlphaFold: {found} ({found/total*100:.1f}%)\n")
            f.write(f"Not in AlphaFold: {not_found}\n")
            f.write(f"No UniProt mapping: {no_mapping}\n")
            f.write(f"API errors: {errors}\n")
            f.write(f"Local mappings: {local_mappings}\n")
            f.write(f"API resolutions: {api_resolutions}\n")
            f.write(f"Parallel workers used: {num_workers}\n\n")
            
            if error_counts:
                f.write("Error breakdown:\n")
                for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {error_type}: {count}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("FOUND IN ALPHAFOLD\n")
            f.write("=" * 70 + "\n")
            f.write("CAFA_ID\tTAXON\tENTRY_NAME\tLOCAL_ACCESSION\tRESOLVED_ACCESSION\tVERSION\tPDB_URL\n")
            for cafa_id, entry_name, local_acc, resolved_acc, version, pdb_url, taxon in sorted(results['found']):
                final_acc = local_acc or resolved_acc or 'N/A'
                f.write(f"{cafa_id}\t{taxon}\t{entry_name}\t{local_acc or 'N/A'}\t{resolved_acc or 'N/A'}\tv{version}\t{pdb_url or 'N/A'}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("NOT FOUND IN ALPHAFOLD\n")
            f.write("=" * 70 + "\n")
            f.write("CAFA_ID\tTAXON\tENTRY_NAME\tLOCAL_ACCESSION\tRESOLVED_ACCESSION\n")
            for cafa_id, entry_name, local_acc, resolved_acc, taxon in sorted(results['not_found']):
                f.write(f"{cafa_id}\t{taxon}\t{entry_name}\t{local_acc or 'N/A'}\t{resolved_acc or 'N/A'}\n")
            
            if results['no_mapping']:
                f.write("\n" + "=" * 70 + "\n")
                f.write("NO UNIPROT MAPPING\n")
                f.write("=" * 70 + "\n")
                for cafa_id in sorted(results['no_mapping']):
                    f.write(f"{cafa_id}\n")
            
            if results['errors']:
                f.write("\n" + "=" * 70 + "\n")
                f.write("API ERRORS\n")
                f.write("=" * 70 + "\n")
                f.write("CAFA_ID\tTAXON\tENTRY_NAME\tLOCAL_ACCESSION\tERROR\n")
                for cafa_id, entry_name, local_acc, error, taxon in sorted(results['errors']):
                    f.write(f"{cafa_id}\t{taxon}\t{entry_name}\t{local_acc or 'N/A'}\t{error}\n")
        
        print(f"\nDetailed results saved to: {output_file}")
    
    return results


def analyze_coverage_by_split(data_dir, results, cafa_to_accession):
    """Analyze AlphaFold coverage breakdown by aspect and split."""
    print("\n" + "="*70)
    print("=== Coverage by Aspect and Split ===")
    print("="*70)
    
    found_set = set(cafa_id for cafa_id, _, _, _, _, _, _ in results['found'])
    
    total_all = 0
    found_all = 0
    
    for aspect in ['BPO', 'CCO', 'MFO']:
        print(f"\n{aspect}:")
        
        for split in ['train', 'valid', 'test']:
            names_file = Path(data_dir) / f"{aspect}_{split}_names.npy"
            
            if not names_file.exists():
                continue
            
            proteins = np.load(names_file, allow_pickle=True)
            n_total = len(proteins)
            n_found = sum(1 for p in proteins if p in found_set)
            n_mappable = sum(1 for p in proteins if p in cafa_to_accession)
            
            total_all += n_total
            found_all += n_found
            
            coverage = n_found / n_total * 100 if n_total > 0 else 0
            coverage_mappable = n_found / n_mappable * 100 if n_mappable > 0 else 0
            
            print(f"  {split:6s}: {n_total:5d} proteins | "
                  f"mappable: {n_mappable:5d} | "
                  f"in AlphaFold: {n_found:5d} ({coverage:.1f}% overall, {coverage_mappable:.1f}% of mappable)")
    
    print(f"\n{'='*70}")
    print(f"Overall: {found_all}/{total_all} proteins ({found_all/total_all*100:.1f}%)")


def download_alphafold_structure(cafa_id, pdb_url, output_dir, session=None):
    """Download AlphaFold PDB file for a protein.
    
    Args:
        cafa_id: Original CAFA protein ID (used for filename)
        pdb_url: Direct URL to PDB file from AlphaFold API
        output_dir: Directory to save PDB files
        session: Optional requests.Session for connection reuse
    
    Returns:
        (success: bool, error: str or None)
    """
    if session is None:
        session = requests.Session()
        close_session = True
    else:
        close_session = False
    
    output_path = Path(output_dir) / f"{cafa_id}.pdb"
    
    # Skip if already downloaded
    if output_path.exists():
        if close_session:
            session.close()
        return True, None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = session.get(pdb_url, timeout=30)
            
            if response.status_code == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                if close_session:
                    session.close()
                return True, None
            
            elif response.status_code == 404:
                if close_session:
                    session.close()
                return False, "PDB file not found"
            
            else:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                if close_session:
                    session.close()
                return False, f"HTTP {response.status_code}"
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            else:
                if close_session:
                    session.close()
                return False, "Timeout"
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            else:
                if close_session:
                    session.close()
                return False, str(e)[:100]
    
    if close_session:
        session.close()
    return False, "Unknown error after retries"


def download_structures_parallel(results, output_dir, num_workers=20):
    """Download all found AlphaFold structures in parallel.
    
    Args:
        results: Results dict from check_alphafold_coverage
        output_dir: Directory to save PDB files
        num_workers: Number of parallel download workers
    """
    print("\n" + "="*70)
    print("=== Downloading AlphaFold Structures ===")
    print("="*70)
    
    found_proteins = results['found']
    
    if not found_proteins:
        print("No proteins found in AlphaFold to download.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check which files already exist
    existing = set()
    to_download = []
    
    for cafa_id, entry_name, local_acc, resolved_acc, version, pdb_url, taxon in found_proteins:
        pdb_path = output_dir / f"{cafa_id}.pdb"
        if pdb_path.exists():
            existing.add(cafa_id)
        else:
            if pdb_url:
                to_download.append((cafa_id, pdb_url))
    
    print(f"Total proteins in AlphaFold: {len(found_proteins)}")
    print(f"Already downloaded: {len(existing)}")
    print(f"To download: {len(to_download)}")
    print(f"Number of workers: {num_workers}")
    
    if not to_download:
        print("\nAll structures already downloaded!")
        return
    
    print(f"\nDownloading to: {output_dir}")
    
    success_count = 0
    error_count = 0
    error_types = defaultdict(int)
    
    def download_single(item):
        cafa_id, pdb_url = item
        session = requests.Session()
        success, error = download_alphafold_structure(cafa_id, pdb_url, output_dir, session)
        session.close()
        return (cafa_id, success, error)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_single, item) for item in to_download]
        
        with tqdm(total=len(futures), desc="Downloading PDB files") as pbar:
            for future in as_completed(futures):
                cafa_id, success, error = future.result()
                
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    if error:
                        error_type = error.split(':')[0] if ':' in error else error
                        error_types[error_type] += 1
                
                pbar.update(1)
    
    print("\n" + "="*70)
    print("=== Download Summary ===")
    print("="*70)
    print(f"Successfully downloaded: {success_count}")
    print(f"Errors: {error_count}")
    
    if error_types:
        print("\nError breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
    
    total_now = len(existing) + success_count
    print(f"\nTotal PDB files available: {total_now}/{len(found_proteins)} "
          f"({total_now/len(found_proteins)*100:.1f}%)")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Check AlphaFold coverage for CAFA3 proteins")
    parser.add_argument("--data-dir", type=str, default="../data", help="Data directory")
    parser.add_argument("--cafa-assessment-dir", type=str,
                        default=os.environ.get("CAFA_ASSESSMENT_DIR", "CAFA_assessment_tool"),
                        help="Path to CAFA assessment tool")
    parser.add_argument("--output-file", type=str, default="../data/alphafold_coverage_results.txt")
    parser.add_argument("--pdb-output-dir", type=str, default="../data/alphafold_structures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cafa_assessment_dir = Path(args.cafa_assessment_dir)
    output_file = Path(args.output_file)
    pdb_output_dir = Path(args.pdb_output_dir)

    # Number of parallel workers
    num_workers_check = 1000  # For checking coverage
    num_workers_download = 20  # For downloading PDB files
    
    print("="*70)
    print("ALPHAFOLD COVERAGE CHECK FOR CAFA3 PROTEINS")
    print("Using existing CAFA ID conversion infrastructure")
    print("="*70)
    
    # Step 1: Build mappings using CAFA infrastructure
    cafa_to_accession = build_cafa_to_accession_mapping(cafa_assessment_dir, data_dir)
    
    # Step 2: Get all CAFA proteins
    all_proteins = get_all_cafa_proteins(data_dir)
    
    # Step 3: Check AlphaFold coverage (parallel)
    results = check_alphafold_coverage(
        all_proteins, 
        cafa_to_accession, 
        output_file=output_file,
        num_workers=num_workers_check
    )
    
    # Step 4: Analyze by split
    analyze_coverage_by_split(data_dir, results, cafa_to_accession)
    
    # Step 5: Download PDB structures
    download_structures_parallel(results, pdb_output_dir, num_workers=num_workers_download)
    
    print("\n" + "="*70)
    print("✓ Done!")
    print("="*70)


if __name__ == "__main__":
    main()