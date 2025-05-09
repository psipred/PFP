#!/bin/bash
# ask for 4 GB of RAM per task, with an upper bound of 4 GB
#$ -l tmem=16G
#$ -l h_vmem=16G



# set maximum runtime to 48 hours
#$ -l h_rt=8:0:0

# merge stdout and stderr to a single output file
#$ -j y
# give the (array) job a name
#$ -N pfam_nw
#$ -t 1-100

#$ -wd /SAN/bioinf/PFP/PFP/cluster/prot2text
#$ -o /SAN/bioinf/PFP/PFP/cluster/prot2text/prot2text.log



# setting a GPU and selection specfic hosts

#$ -l gpu=true
#$ -pe gpu 1

# # print the location that task is running (helpful for debugging)
hostname

# Source the conda initialization script
# source /scratch0/miniconda3/etc/profile.d/conda.sh



# Load your Python environment (modify based on your setup)
# source /share/apps/source_files/python-3.8.5.source

# Run the script

source /SAN/bioinf/PFP/conda/miniconda3/etc/profile.d/conda.sh
conda activate pfp

python /SAN/bioinf/PFP/PFP/cluster/prot2text/prot2text_agent.py

# python -c "
# from cluster.esm_residue import generate_esm_embeddings
# generate_esm_embeddings(
#   fasta_file='/SAN/bioinf/PFP/dataset/CAFA3/CAFA3_training_data/uniprot_sprot_exp.fasta',
#   model_name='facebook/esm1b_t33_650M_UR50S',
#   output_dir='embeddings/cafa/esm',
#   batch_size=32,
#   max_length=1022,
#   use_gpu=True
# )
# "

