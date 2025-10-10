#!/usr/bin/env python3
"""
GO Annotation Pipeline for Protein Dataset
Filters proteins by length, retrieves experimental GO annotations,
applies GO hierarchy propagation with official exclusion lists.
"""

import os
import gzip
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Tuple

# Configuration
INPUT_FASTA = "/home/zijianzhou/Datasets/protad/protad.fasta"
OUTPUT_DIR = "/home/zijianzhou/Datasets/protad/go_annotations"
MAX_SEQUENCE_LENGTH = 1024

# URLs for data sources
GOA_GAF_URL = "http://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz"
GO_OBO_URL = "http://current.geneontology.org/ontology/go-basic.obo"
GO_EXCLUDE_DO_NOT_ANNOTATE_URL = "https://current.geneontology.org/ontology/subsets/gocheck_do_not_annotate.obo"
GO_EXCLUDE_OBSOLETION_URL = "https://current.geneontology.org/ontology/subsets/gocheck_obsoletion_candidate.obo"

# Experimental evidence codes
EXPERIMENTAL_EVIDENCE_CODES = {
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP',  # Manual experimental
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'          # High-throughput experimental
}

# GO root terms to exclude
GO_ROOT_TERMS = {
    'GO:0008150',  # biological_process
    'GO:0003674',  # molecular_function
    'GO:0005575'   # cellular_component
}


def load_fasta(filepath: str) -> Dict[str, str]:
    """Load FASTA file and return dict of protein_id -> sequence."""
    print(f"Loading FASTA from {filepath}...")
    proteins = {}
    current_id = None
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    proteins[current_id] = ''.join(current_seq)
                # Extract UniProt ID from header (format: >sp|P12345|NAME ...)
                parts = line[1:].split('|')
                if len(parts) >= 2:
                    current_id = parts[1]  # UniProt accession
                else:
                    current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id:
            proteins[current_id] = ''.join(current_seq)
    
    print(f"Loaded {len(proteins)} proteins")
    return proteins


def filter_by_length(proteins: Dict[str, str], max_length: int) -> Dict[str, str]:
    """Filter proteins by sequence length."""
    print(f"Filtering proteins with length <= {max_length}...")
    filtered = {pid: seq for pid, seq in proteins.items() if len(seq) <= max_length}
    print(f"Retained {len(filtered)} proteins (removed {len(proteins) - len(filtered)})")
    return filtered


def download_file(url: str, filepath: str):
    """Download file if it doesn't exist."""
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return
    
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"Downloaded to {filepath}")


def parse_obo_file(filepath: str) -> Set[str]:
    """Parse OBO file and extract GO term IDs."""
    go_terms = set()
    current_term = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('id: GO:'):
                current_term = line.split('id: ')[1]
                go_terms.add(current_term)
    
    return go_terms


def parse_go_ontology(filepath: str) -> Tuple[Dict[str, Set[str]], Dict[str, Dict]]:
    """Parse GO OBO file and build hierarchy (child -> parents) including part_of relationships."""
    print("Parsing GO ontology...")
    child_to_parents = defaultdict(set)
    term_info = {}
    
    current_term = None
    current_name = None
    current_namespace = None
    is_obsolete = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line == '[Term]':
                if current_term and not is_obsolete:
                    term_info[current_term] = {
                        'name': current_name,
                        'namespace': current_namespace
                    }
                current_term = None
                current_name = None
                current_namespace = None
                is_obsolete = False
                
            elif line.startswith('id: GO:'):
                current_term = line.split('id: ')[1]
                
            elif line.startswith('name: '):
                current_name = line.split('name: ')[1]
                
            elif line.startswith('namespace: '):
                current_namespace = line.split('namespace: ')[1]
                
            elif line.startswith('is_a: GO:'):
                parent = line.split('is_a: ')[1].split(' ! ')[0]
                if current_term:
                    child_to_parents[current_term].add(parent)
            
            elif line.startswith('relationship: part_of GO:'):
                # Parse part_of relationships
                parent = line.split('relationship: part_of ')[1].split(' ! ')[0]
                if current_term:
                    child_to_parents[current_term].add(parent)
                    
            elif line.startswith('is_obsolete: true'):
                is_obsolete = True
    
    # Add last term
    if current_term and not is_obsolete:
        term_info[current_term] = {
            'name': current_name,
            'namespace': current_namespace
        }
    
    print(f"Parsed {len(term_info)} GO terms with hierarchy")
    return child_to_parents, term_info


def parse_gaf_file(filepath: str, target_proteins: Set[str]) -> Dict[str, Set[str]]:
    """Parse GAF file and extract experimental GO annotations, excluding negated/ambiguous annotations."""
    print("Parsing GAF file for experimental annotations...")
    protein_go = defaultdict(set)
    
    if filepath.endswith('.gz'):
        f = gzip.open(filepath, 'rt')
    else:
        f = open(filepath, 'r')
    
    try:
        for line in f:
            if line.startswith('!'):
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 15:
                continue
            
            db_object_id = fields[1]  # UniProt ID
            qualifier = fields[3]      # Qualifier (e.g., NOT, contributes_to, colocalizes_with)
            go_id = fields[4]
            evidence_code = fields[6]
            
            qset = set(qualifier.split('|')) if qualifier else set()
            if {'NOT', 'contributes_to', 'colocalizes_with'} & qset:
                continue

            
            # Filter: only target proteins and experimental evidence
            if db_object_id in target_proteins and evidence_code in EXPERIMENTAL_EVIDENCE_CODES:
                protein_go[db_object_id].add(go_id)
    
    finally:
        f.close()
    
    print(f"Found experimental annotations for {len(protein_go)} proteins")
    return protein_go


def propagate_go_terms(protein_go: Dict[str, Set[str]], 
                       child_to_parents: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Propagate GO terms up the hierarchy."""
    print("Propagating GO terms to ancestors...")
    propagated = {}
    
    for protein_id, go_terms in protein_go.items():
        all_terms = set(go_terms)
        
        # For each GO term, add all ancestors
        for go_term in go_terms:
            visited = set()
            queue = [go_term]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                all_terms.add(current)
                
                # Add parents to queue
                if current in child_to_parents:
                    for parent in child_to_parents[current]:
                        if parent not in visited:
                            queue.append(parent)
        
        propagated[protein_id] = all_terms
    
    return propagated


def apply_exclusions(protein_go: Dict[str, Set[str]], 
                     excluded_terms: Set[str]) -> Dict[str, Set[str]]:
    """Remove excluded GO terms (roots + exclusion lists) from annotations."""
    all_excluded = excluded_terms | GO_ROOT_TERMS
    print(f"Applying exclusions ({len(all_excluded)} terms to exclude: {len(excluded_terms)} from lists + {len(GO_ROOT_TERMS)} roots)...")
    filtered = {}
    
    for protein_id, go_terms in protein_go.items():
        valid_terms = go_terms - all_excluded
        if valid_terms:  # Only keep proteins with valid GO terms
            filtered[protein_id] = valid_terms
    
    print(f"Retained {len(filtered)} proteins after exclusion")
    return filtered


def save_fasta(proteins: Dict[str, str], protein_ids: Set[str], filepath: str):
    """Save filtered proteins to FASTA file."""
    print(f"Saving {len(protein_ids)} proteins to {filepath}...")
    with open(filepath, 'w') as f:
        for protein_id in sorted(protein_ids):
            if protein_id in proteins:
                f.write(f">sp|{protein_id}|\n")
                seq = proteins[protein_id]
                # Write sequence in 60 character lines
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i+60] + '\n')


def save_annotations(protein_go: Dict[str, Set[str]], 
                     term_info: Dict[str, Dict],
                     output_dir: str):
    """Save GO annotations and metadata."""
    # Save protein -> GO mappings
    annotations_file = os.path.join(output_dir, "protein_go_annotations.tsv")
    print(f"Saving annotations to {annotations_file}...")
    with open(annotations_file, 'w') as f:
        f.write("protein_id\tgo_terms\n")
        for protein_id in sorted(protein_go.keys()):
            go_list = ','.join(sorted(protein_go[protein_id]))
            f.write(f"{protein_id}\t{go_list}\n")
    
    # Count GO term frequencies
    go_counts = defaultdict(int)
    for go_terms in protein_go.values():
        for go_term in go_terms:
            go_counts[go_term] += 1
    
    # Save GO term info
    go_info_file = os.path.join(output_dir, "go_terms_info.tsv")
    print(f"Saving GO term info to {go_info_file}...")
    with open(go_info_file, 'w') as f:
        f.write("go_id\tname\taspect\tcount\n")
        for go_id in sorted(go_counts.keys()):
            info = term_info.get(go_id, {'name': 'Unknown', 'namespace': 'Unknown'})
            f.write(f"{go_id}\t{info['name']}\t{info['namespace']}\t{go_counts[go_id]}\n")


def save_statistics(stats: Dict, filepath: str):
    """Save statistics as JSON."""
    print(f"Saving statistics to {filepath}...")
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    stats = {
        'initial_proteins': 0,
        'after_length_filter': 0,
        'with_experimental_annotations': 0,
        'after_propagation': 0,
        'after_exclusion_filter': 0,
        'final_proteins': 0,
        'evidence_codes_used': list(EXPERIMENTAL_EVIDENCE_CODES)
    }
    
    # Step 1: Load FASTA
    proteins = load_fasta(INPUT_FASTA)
    stats['initial_proteins'] = len(proteins)
    
    # Step 2: Filter by length
    proteins = filter_by_length(proteins, MAX_SEQUENCE_LENGTH)
    stats['after_length_filter'] = len(proteins)
    
    # Step 3: Download data files
    gaf_file = os.path.join(OUTPUT_DIR, "goa_uniprot_all.gaf.gz")
    obo_file = os.path.join(OUTPUT_DIR, "go-basic.obo")
    exclude_do_not_annotate = os.path.join(OUTPUT_DIR, "gocheck_do_not_annotate.obo")
    exclude_obsoletion = os.path.join(OUTPUT_DIR, "gocheck_obsoletion_candidate.obo")
    
    download_file(GOA_GAF_URL, gaf_file)
    download_file(GO_OBO_URL, obo_file)
    download_file(GO_EXCLUDE_DO_NOT_ANNOTATE_URL, exclude_do_not_annotate)
    download_file(GO_EXCLUDE_OBSOLETION_URL, exclude_obsoletion)
    
    # Step 4: Parse exclusion lists
    print("Loading exclusion lists...")
    excluded_terms = set()
    excluded_terms.update(parse_obo_file(exclude_do_not_annotate))
    excluded_terms.update(parse_obo_file(exclude_obsoletion))
    print(f"Total excluded terms: {len(excluded_terms)}")
    
    # Save excluded terms (including roots)
    all_excluded = excluded_terms | GO_ROOT_TERMS
    with open(os.path.join(OUTPUT_DIR, "excluded_go_terms.txt"), 'w') as f:
        f.write("# GO Root Terms\n")
        for term in sorted(GO_ROOT_TERMS):
            f.write(f"{term}\n")
        f.write("\n# Terms from Exclusion Lists\n")
        for term in sorted(excluded_terms):
            f.write(f"{term}\n")
    
    # Step 5: Parse GO ontology
    child_to_parents, term_info = parse_go_ontology(obo_file)
    
    # Step 6: Parse GAF for experimental annotations
    target_proteins = set(proteins.keys())
    protein_go = parse_gaf_file(gaf_file, target_proteins)
    stats['with_experimental_annotations'] = len(protein_go)
    
    # Step 7: Propagate GO terms
    protein_go = propagate_go_terms(protein_go, child_to_parents)
    stats['after_propagation'] = len(protein_go)
    
    # Step 8: Apply exclusions
    protein_go = apply_exclusions(protein_go, excluded_terms)
    stats['after_exclusion_filter'] = len(protein_go)
    
    # Step 9: Final protein set (intersection of length-filtered and annotated)
    final_proteins = set(protein_go.keys())
    stats['final_proteins'] = len(final_proteins)
    
    # Calculate GO statistics
    go_by_aspect = defaultdict(set)
    total_annotations = 0
    for go_terms in protein_go.values():
        total_annotations += len(go_terms)
        for go_term in go_terms:
            if go_term in term_info:
                aspect = term_info[go_term]['namespace']
                go_by_aspect[aspect].add(go_term)
    
    stats['total_go_terms'] = len(set().union(*protein_go.values()))
    stats['go_terms_by_aspect'] = {k: len(v) for k, v in go_by_aspect.items()}
    stats['avg_go_terms_per_protein'] = total_annotations / len(protein_go) if protein_go else 0
    stats['excluded_terms_count'] = len(excluded_terms)
    stats['excluded_root_terms'] = len(GO_ROOT_TERMS)
    stats['total_excluded'] = len(excluded_terms) + len(GO_ROOT_TERMS)
    
    # Step 10: Save outputs
    save_fasta(proteins, final_proteins, os.path.join(OUTPUT_DIR, "proteins_filtered.fasta"))
    save_annotations(protein_go, term_info, OUTPUT_DIR)
    save_statistics(stats, os.path.join(OUTPUT_DIR, "filtering_statistics.json"))
    
    # Save evidence codes used
    with open(os.path.join(OUTPUT_DIR, "evidence_codes_used.txt"), 'w') as f:
        for code in sorted(EXPERIMENTAL_EVIDENCE_CODES):
            f.write(f"{code}\n")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Initial proteins:                 {stats['initial_proteins']:,}")
    print(f"After length filter (≤{MAX_SEQUENCE_LENGTH}):      {stats['after_length_filter']:,}")
    print(f"With experimental annotations:    {stats['with_experimental_annotations']:,}")
    print(f"After propagation:                {stats['after_propagation']:,}")
    print(f"After exclusion filter:           {stats['after_exclusion_filter']:,}")
    print(f"Final proteins:                   {stats['final_proteins']:,}")
    print(f"\nTotal unique GO terms:            {stats['total_go_terms']:,}")
    print(f"Avg GO terms per protein:         {stats['avg_go_terms_per_protein']:.1f}")
    print(f"\nGO terms by aspect:")
    for aspect, count in stats['go_terms_by_aspect'].items():
        print(f"  {aspect:25} {count:,}")
    print(f"\nExcluded terms:")
    print(f"  Root terms:                   {stats['excluded_root_terms']}")
    print(f"  Exclusion list terms:         {stats['excluded_terms_count']}")
    print(f"  Total excluded:               {stats['total_excluded']}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()