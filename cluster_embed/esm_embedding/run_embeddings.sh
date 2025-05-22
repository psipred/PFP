#!/bin/bash
#$ -N esm_embeddings
#$ -l h_vmem=8G
#$ -l tmem=8G
#$ -l h_rt=20:0:0
#$ -j y
#$ -o /SAN/bioinf/PFP/PFP/cluster/esm_embedding/logs/esm_embedding.log
#$ -wd /SAN/bioinf/PFP/PFP
#$ -l gpu=true
#$ -pe gpu 1


pwd
# Activate the env
source /SAN/bioinf/PFP/conda/miniconda3/etc/profile.d/conda.sh
conda activate pfp

# Make sure conda's C++ runtime comes first
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# cafa 3
# python -c "
# from cluster.esm_embedding.esm_residue import generate_esm_embeddings
# generate_esm_embeddings(
#   fasta_file='/SAN/bioinf/PFP/dataset/CAFA3/CAFA3_training_data/uniprot_sprot_exp.fasta',
#   model_name='facebook/esm1b_t33_650M_UR50S',
#   output_dir='/SAN/bioinf/PFP/embeddings/esm',
#   batch_size=16,
#   max_length=1022,
#   use_gpu=True
# )
# "

# cafa 5 small
python -c "
from cluster.esm_embedding.esm_residue import generate_esm_embeddings
generate_esm_embeddings(
  fasta_file='/SAN/bioinf/PFP/dataset/CAFA5_small/filtered_train_seq.fasta',
  model_name='facebook/esm1b_t33_650M_UR50S',
  output_dir='/SAN/bioinf/PFP/embeddings/cafa5_small/esm',
  batch_size=16,
  max_length=1022,
  use_gpu=True
)
"


