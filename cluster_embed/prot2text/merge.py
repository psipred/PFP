import json
import glob
import os

partial_dir = "/SAN/bioinf/PFP/embeddings/cafa5_small/prot2text/temp/"
merged_output_path = "/SAN/bioinf/PFP/embeddings/cafa5_small/prot2text/text/generated_desc.json"

# 1) Load existing entries from generated_desc.json (if it exists)
if os.path.isfile(merged_output_path):
    with open(merged_output_path, "r") as f:
        merged_dict = json.load(f)
    print(f"Loaded {len(merged_dict)} existing records from {merged_output_path}")
else:
    merged_dict = {}
    print("No existing generated_desc.json found, starting with an empty dictionary.")

# 2) Merge in the partial files
all_partials = glob.glob(partial_dir + "partial_output_*.json")
for file_path in all_partials:
    with open(file_path, "r") as f:
        data = json.load(f)
    # Merge into the main dictionary, skipping duplicates
    for seq_id, desc in data.items():
        if seq_id not in merged_dict:
            merged_dict[seq_id] = desc

# 3) Write combined data back to generated_desc.json
with open(merged_output_path, "w") as f:
    json.dump(merged_dict, f, indent=2)

print(f"Merged {len(all_partials)} partial files. Final count: {len(merged_dict)} records in {merged_output_path}")