#!/bin/bash
# Master script to submit experiments

SCRIPT_DIR="/SAN/bioinf/PFP/PFP/structure/experiment_pipeline/scripts"

# Priority experiments (run first)
PRIORITY_EXPS=(
    "graph_knn_k5"
    "graph_knn_k10"
    "graph_knn_k15"
    "graph_knn_k20"
    "graph_radius_r8"
    "graph_radius_r10"
    "graph_radius_r12"
    "model_layers_2"
    "model_layers_4"
    "model_layers_6"
    "model_hidden_256"
    "model_hidden_512"


)


echo "Submitting priority experiments..."
for exp in "${PRIORITY_EXPS[@]}"; do
    if [ -f "$SCRIPT_DIR/${exp}.sh" ]; then
        echo "Submitting: $exp"
        qsub "$SCRIPT_DIR/${exp}.sh"
        sleep 1
    fi
done

echo "Priority experiments submitted."
echo "To submit remaining experiments, use:"
echo "  for script in $SCRIPT_DIR/*.sh; do qsub \$script; sleep 60; done"
