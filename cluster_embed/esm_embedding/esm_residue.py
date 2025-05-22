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
    pad_to_length: int = 1022,
    use_gpu: bool = True,
    mean_pool: bool = True,
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
        pad_to_length (int): When mean_pool is False, pad/truncate residue dimension to this length.
        use_gpu (bool): Whether to run inference on GPU if available.
        mean_pool (bool): If True save a single 1280‑D vector (mean of residues).
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
        padding_mode = "longest" if mean_pool else "max_length"
        encoded = tokenizer(
            batch_strs,
            padding=padding_mode,
            truncation=True,              # always truncate at model limit
            max_length=pad_to_length + 2, # 1022 residues + BOS + EOS
            return_tensors="pt",
            add_special_tokens=True,
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
            mask = encoded["attention_mask"][idx].bool()          # 1 for BOS/residues/EOS
            emb  = outputs[0][idx].cpu()                          # [L, D]
            # find last true position (EOS), drop BOS/EOS
            last_idx = mask.nonzero()[-1].item()
            residues = emb[1:last_idx]                            # (seq_len, D)

            if mean_pool:
                saved_emb = residues.mean(dim=0).numpy().astype(np.float32)      # (D,)
            else:
                # Pad / truncate to fixed length
                if residues.size(0) < pad_to_length:
                    pad_rows = pad_to_length - residues.size(0)
                    pad = torch.zeros(pad_rows, residues.size(1))
                    residues = torch.cat([residues, pad], dim=0)
                else:
                    residues = residues[:pad_to_length]
                saved_emb = residues.numpy().astype(np.float32)   # (pad_to_length, D)


            np.save(
                os.path.join(output_dir, f"{seq_id}.npy"),
                {"name": seq_id, "embedding": saved_emb},
                allow_pickle=True,
            )

    print(f"Done! Generated embeddings are saved under '{output_dir}'.")


# cafa 3
# """
# python -c "
# from cluster.esm_embedding.esm_residue import generate_esm_embeddings
# generate_esm_embeddings(
#   fasta_file='/SAN/bioinf/PFP/dataset/CAFA3/CAFA3_training_data/uniprot_sprot_exp.fasta',
#   model_name='facebook/esm1b_t33_650M_UR50S',
#   output_dir='/SAN/bioinf/PFP/embeddings/esm',
#   batch_size=16,
#   pad_to_length=1022,
#   use_gpu=True
# )
# "
# """

# cafa 5_small 

#mmsite embedding with fixed length
"""
python -c "
from cluster_embed.esm_embedding.esm_residue import generate_esm_embeddings
generate_esm_embeddings(
  fasta_file='/SAN/bioinf/PFP/dataset/CAFA5_small/filtered_train_seq.fasta',
  model_name='facebook/esm1b_t33_650M_UR50S',
  output_dir='/SAN/bioinf/PFP/embeddings/cafa5_small/esm',
  batch_size=16,
  mean_pool=False,      # keep per-residue
  pad_to_length=1022,
  use_gpu=True
)
"
"""

# mean pooling embeddings 
"""
python -c "
from cluster_embed.esm_embedding.esm_residue import generate_esm_embeddings
generate_esm_embeddings(
  fasta_file='/SAN/bioinf/PFP/dataset/CAFA5_small/filtered_train_seq.fasta',
  model_name='facebook/esm1b_t33_650M_UR50S',
  output_dir='/SAN/bioinf/PFP/embeddings/cafa5_small/esm_mean',
  batch_size=32,
  mean_pool=True,      # keep per-residue
  use_gpu=True
)
"
"""