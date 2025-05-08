import sys
sys.path.append("/SAN/bioinf/PFP")
from Prot2Text.prot2text_model.Model import Prot2TextModel
from Prot2Text.prot2text_model.tokenization_prot2text import Prot2TextTokenizer
import torch
from Bio import SeqIO
import json
import os
import math


# Initialize paths and device
model_path = '/SAN/bioinf/PFP/pretrained/esm2text_base'
# fasta_path = '/SAN/bioinf/PFP/scratch/filtered_train_seq.fasta'
fasta_path = '/SAN/bioinf/PFP/dataset/CAFA3/CAFA3_training_data/uniprot_sprot_exp.fasta'
output_json_path = '/SAN/bioinf/PFP/embeddings/prot2text/cafa/generated_desc.json'



# Get SGE_TASK_ID from environment; if not running in array mode, default to 1.
sge_task_id = int(os.environ.get("SGE_TASK_ID", 1))
# Define total number of array tasks (if not set as an env variable, hardcode it based on your -t range)
total_tasks = int(os.environ.get("SGE_TASK_LAST", 1))

print(f"SGE_TASK_ID: {sge_task_id}")

# Load existing JSON if present
try:
    with open(output_json_path) as f:
        entry_prot2text = json.load(f)
except FileNotFoundError:
    entry_prot2text = {}


# Print total number of records in the FASTA file
total_records = sum(1 for _ in SeqIO.parse(fasta_path, 'fasta'))
print(f"Total number of records in the FASTA file: {total_records}")


# Print how many are already processed
already_processed = len(entry_prot2text)
print(f"Already processed: {already_processed}")

# How many are left
remaining_to_process = total_records - already_processed
print(f"Remaining to process: {remaining_to_process}")

# Compute chunk size and determine start and end indices for this task
chunk_size = math.ceil(total_records / total_tasks)
start_index = (sge_task_id - 1) * chunk_size
end_index = min(sge_task_id * chunk_size, total_records)
print(f"Processing records from index {start_index} to {end_index} (Task {sge_task_id})")


save_interval = 1  # define how many records to process before saving
force_regeneration = False

# Set up the device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Load model and tokenizer
model = Prot2TextModel.from_pretrained(model_path)
tokenizer = Prot2TextTokenizer.from_pretrained(model_path)

@torch.no_grad()
def prot2text(seq):
    """
    Generates a description text for a given protein sequence
    using the loaded Prot2Text model and tokenizer.
    """
    # Note: "generate_protein_description" returns a list of strings;
    #   if it returns more than one, adjust accordingly
    description = model.generate_protein_description(
        protein_sequence=seq,
        tokenizer=tokenizer,
        device=device
    )
    return description



processed_records = 0
local_results = {}
for i, record in enumerate(SeqIO.parse(fasta_path, 'fasta')):
    # Only process records within the [start_index, end_index) range for this task
    if i < start_index or i >= end_index:
        continue

    # If the record is already processed and not forced to regenerate, skip it.
    if record.id in entry_prot2text and not force_regeneration:
        continue

    # print(f"Processing record {i} with ID: {record.id}")
    seq = str(record.seq)
    text = prot2text(seq)
    if isinstance(text, list):
        text = text[0] if len(text) > 0 else ""
    local_results[record.id] = text
    processed_records += 1

    if processed_records % save_interval == 0:
        with open(f"/SAN/bioinf/PFP/embeddings/prot2text/cafa/temp/partial_output_{sge_task_id}.json", 'w') as f:
            json.dump(local_results, f, indent=2)
        print(f"Partial save: {processed_records} records processed (Task {sge_task_id}).")


exit()
with open(f"/SAN/bioinf/PFP/scratch/testdata/partial/partial_output_{sge_task_id}.json", 'w') as f:
    json.dump(entry_prot2text, f, indent=2)
print(f"Finished processing. Partial results for task {sge_task_id} saved to: testdata/partial_output_{sge_task_id}.json")
