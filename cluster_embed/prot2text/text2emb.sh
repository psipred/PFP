#!/bin/bash
#$ -N text2emb
#$ -t 1-100
#$ -l h_vmem=16G
#$ -l tmem=16G
#$ -l h_rt=4:0:0
#$ -j y
#$ -o /SAN/bioinf/PFP/PFP/cluster_embed/prot2text/logs/text2emb.log
#$ -wd /SAN/bioinf/PFP/PFP/cluster_embed/prot2text
#$ -l gpu=true
#$ -pe gpu 1

# Activate conda environment
source /SAN/bioinf/PFP/conda/miniconda3/etc/profile.d/conda.sh
conda activate pfp-gpu
# Make sure the env’s newer C++ runtime is used


# (optional sanity checks)
which python   # → …/envs/pfp/bin/python
python --version


# Directory containing JSON files
# dir_json="/SAN/bioinf/PFP/embeddings/prot2text/cafa/temp"
dir_json="/SAN/bioinf/PFP/embeddings/cafa5_small/prot2text/temp"
json_files=($dir_json/*.json)

# Get the JSON file for this task
json_file=${json_files[$((SGE_TASK_ID-1))]}

# Output directory for this file
# output_dir="/SAN/bioinf/PFP/embeddings/prot2text/cafa/text_embeddings"
output_dir="/SAN/bioinf/PFP/embeddings/cafa5_small/prot2text/text_embeddings"



echo "Processing $json_file on $HOSTNAME, output to $output_dir"

python /SAN/bioinf/PFP/PFP/cluster_embed/prot2text/prot2text.py \
    --json_file "$json_file" \
    --output_dir "$output_dir" \
    --use_gpu True 


python /SAN/bioinf/PFP/PFP/cluster_embed/prot2text/prot2text.py \
    --json_file "/SAN/bioinf/PFP/embeddings/cafa5_small/prot2text/temp/partial_output_1.json" \
    --output_dir "/SAN/bioinf/PFP/embeddings/cafa5_small/prot2text/test" \
    --use_gpu True 

