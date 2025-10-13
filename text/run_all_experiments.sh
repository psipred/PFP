
#!/bin/bash

# Run all experiments: 3 aspects × 4 thresholds = 12 experiments

for aspect in BP MF CC; do
    for threshold in 30 50 70 95; do
        echo "Running: Aspect=$aspect, Threshold=$threshold"
        python train.py --aspect $aspect --threshold $threshold --model esm 
    done
done

echo "All experiments complete!"