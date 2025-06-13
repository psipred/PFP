#!/bin/bash
#$ -N graph_radius_r10
#$ -l h_vmem=40G
#$ -l tmem=40G
#$ -l h_rt=8:0:0
#$ -j y
#$ -o /SAN/bioinf/PFP/PFP/structure/experiments/logs/graph_radius_r10.log
#$ -wd /SAN/bioinf/PFP/PFP/structure
#$ -l gpu=true
#$ -l h=!walter*
#$ -pe gpu 1

echo "Starting experiment: graph_radius_r10"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Activate environment
source /SAN/bioinf/PFP/conda/miniconda3/etc/profile.d/conda.sh
conda activate train

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run training
cd /SAN/bioinf/PFP/PFP/structure/experiment_pipeline
python train_with_metrics.py --config-path=./configs --config-name=graph_radius_r10

echo "Experiment completed: $(date)"
