#!/bin/bash
#$ -N model_hidden_512
#$ -l h_vmem=40G
#$ -l tmem=40G
#$ -l h_rt=8:0:0
#$ -j y
#$ -o /SAN/bioinf/PFP/PFP/structure/experiments/logs/model_hidden_512.log
#$ -wd /SAN/bioinf/PFP/PFP/structure
#$ -l gpu=true
#$ -l h=!walter*
#$ -pe gpu 1

echo "Starting experiment: model_hidden_512"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Activate environment
source /SAN/bioinf/PFP/conda/miniconda3/etc/profile.d/conda.sh
conda activate train

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run training
cd /SAN/bioinf/PFP/PFP/structure/experiment_pipeline
python train_with_metrics.py --config-path=./configs --config-name=model_hidden_512

echo "Experiment completed: $(date)"
