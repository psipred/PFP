import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def generate_biomed_text_embeddings(
    json_file: str,
    model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    output_dir: str = "testdata/biomed_text_embeddings",
    batch_size: int = 16,
    max_length: int = 128,
    use_gpu: bool = True
):
    """
    Reads a JSON file mapping protein IDs to text descriptions, tokenizes each text
    using the specified HuggingFace model/tokenizer, performs a forward pass to obtain
    embeddings, and saves the per-protein averaged embedding (across token dimension) as a .npy file.

    Args:
        json_file (str): Path to the JSON file containing protein ID and text pairs.
        model_name (str): HuggingFace checkpoint name for BiomedBERT.
        output_dir (str): Directory to save .npy embedding files.
        batch_size (int): Number of text descriptions to process in a single batch.
        max_length (int): Maximum token length for text inputs (truncated/padded).
        use_gpu (bool): Whether to run inference on GPU if available.
    """
    # -------------------- 1. Setup device, load tokenizer & model --------------------
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # -------------------- 2. Load the JSON file and check which IDs to process --------------------
    with open(json_file, 'r') as f:
        prior_knowledge = json.load(f)

    texts_to_process = []
    for protein_id, text_str in prior_knowledge.items():
        out_file = os.path.join(output_dir, f"{protein_id}.npy")
        if os.path.isfile(out_file):
            # Skip if already processed
            continue
        texts_to_process.append((protein_id, text_str))
    
    if not texts_to_process:
        print(f"All embeddings for '{json_file}' already exist in '{output_dir}'. Nothing to do.")
        return

    # -------------------- 3. Process texts in batches --------------------
    for i in tqdm(range(0, len(texts_to_process), batch_size)):
        batch = texts_to_process[i : i + batch_size]
        batch_ids = [pid for (pid, _) in batch]
        batch_texts = [text for (_, text) in batch]

        # 3A) Tokenize the batch of texts.
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # 3B) Run a forward pass through the model.
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs[0] has shape [batch_size, seq_len, hidden_dim]
            token_embeddings = outputs[0]

        # 3C) Pool embeddings: average over the token dimension.
        # This yields a [batch_size, hidden_dim] tensor.
        embeddings = token_embeddings.cpu().numpy()
        
        # -------------------- 4. Save embeddings --------------------
        for idx, protein_id in enumerate(batch_ids):
            emb = embeddings[idx]
            out_file = os.path.join(output_dir, f"{protein_id}.npy")
            np.save(out_file, {"name": protein_id, "embedding": emb}, allow_pickle=True)

    print(f"Done! Generated embeddings are saved under '{output_dir}'.")


# Example command line usage:
if __name__ == "__main__":
    generate_biomed_text_embeddings(
        json_file="testdata/generated_desc.json",
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        output_dir="testdata/biomed_text_embeddingss",
        batch_size=16,
        max_length=128,
        use_gpu=True
    )
