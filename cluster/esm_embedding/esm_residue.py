import os
import torch
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel

def generate_esm_embeddings(
    fasta_file: str,
    model_name: str = "facebook/esm1b_t33_650M_UR50S",
    output_dir: str = "./esm_embeddings",
    batch_size: int = 16,
    max_length: int = 1022,
    use_gpu: bool = True
):
    """
    Reads a FASTA file of (id, sequence) pairs, tokenizes each sequence using
    the specified HuggingFace ESM model/tokenizer, runs a forward pass, and
    saves the per-residue embedding (model(...)[0]) as a .npy file.

    Args:
        fasta_file (str): Path to the FASTA file, each entry has ID and sequence.
        model_name (str): HuggingFace model checkpoint name.
        output_dir (str): Where to save .npy embedding files.
        batch_size (int): Number of sequences per batch.
        max_length (int): Sequence length to truncate/pad to.
        use_gpu (bool): Whether to run inference on GPU if available.
    """

    # -------------------- 1. Setup device, load tokenizer & model --------------------
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    # If you prefer to keep everything on CPU for debugging, uncomment:

    print(device)
    # exit()
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        

    # -------------------- 2. Read FASTA; skip IDs if embedding already exists --------------------
    sequences_to_process = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_id = record.id
        out_file = os.path.join(output_dir, f"{seq_id}.npy")
        if os.path.isfile(out_file):
            # Skip this sequence if the file already exists
            continue
        seq_str = str(record.seq)
        sequences_to_process.append((seq_id, seq_str))

    # If everything is already processed, we can just exit early
    if not sequences_to_process:
        print(f"All embeddings for '{fasta_file}' already exist in '{output_dir}'. Nothing to do.")
        return

    # -------------------- 3. Iterate over sequences in batches --------------------
    for i in tqdm(range(0, len(sequences_to_process), batch_size)):
        batch_data = sequences_to_process[i : i + batch_size]
        batch_ids = [seq_id for (seq_id, _) in batch_data]
        batch_strs = [seq_str for (_, seq_str) in batch_data]

        # Tokenize
        encoded = tokenizer(
            batch_strs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs[0] has shape [batch_size, seq_len, hidden_dim]

        # Move to CPU and convert to NumPy
        embeddings = outputs[0].cpu().numpy()

        # -------------------- 4. Save each sequence embedding --------------------
        for idx, seq_id in enumerate(batch_ids):
            arr = embeddings[idx]  # shape: [seq_len, hidden_dim]
            out_file = os.path.join(output_dir, f"{seq_id}.npy")
            np.save(out_file, {"name": seq_id, "embedding": arr}, allow_pickle=True)

    print(f"Done! Generated embeddings are saved under '{output_dir}'.")

"""
python -c "
from cluster.esm_embedding.esm_residue import generate_esm_embeddings
generate_esm_embeddings(
  fasta_file='/SAN/bioinf/PFP/dataset/CAFA3/CAFA3_training_data/uniprot_sprot_exp.fasta',
  model_name='facebook/esm1b_t33_650M_UR50S',
  output_dir='/SAN/bioinf/PFP/embeddings/esm',
  batch_size=16,
  max_length=1022,
  use_gpu=True
)
"
"""
