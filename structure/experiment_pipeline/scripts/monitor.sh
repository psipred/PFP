#!/bin/bash
echo "Structure GO Prediction Experiments Status"
echo "=========================================="

# Running jobs
echo -e "\nRunning:"
qstat | grep -E "graph_|model_|train_|bpo_|cco_|mfo_" | awk '{print $1, $3, $5, $8}'

# Completed experiments
echo -e "\nCompleted:"
find /SAN/bioinf/PFP/PFP/structure/experiments -name "training_summary.json" | while read f; do
    exp=$(basename $(dirname $f))
    echo "  - $exp"
done

# Failed experiments
echo -e "\nFailed (check logs):"
grep -l "Error\|Traceback" /SAN/bioinf/PFP/PFP/structure/experiments/logs/*.log 2>/dev/null | while read f; do
    echo "  - $(basename $f .log)"
done
