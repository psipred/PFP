#!/bin/bash

# Run all experiments: 3 aspects × 4 thresholds × 4 models = 48 experiments

for aspect in  MF; do
    for threshold in 50; do
        echo "Running: Aspect=$aspect, Threshold=$threshold"
        
        # # Run ESM baseline
        python train.py --aspect $aspect --threshold $threshold --model esm
        
        # # Run Function-only model
        python train.py --aspect $aspect --threshold $threshold --model function
        
        # Run Concat baseline
        python train.py --aspect $aspect --threshold $threshold --model concat
        
        # Run Text fusion model
        python train.py --aspect $aspect --threshold $threshold --model text

        # Run Gated fusion model
        # python train.py --aspect $aspect --threshold $threshold --model gated

    done
done

echo "All experiments complete!"