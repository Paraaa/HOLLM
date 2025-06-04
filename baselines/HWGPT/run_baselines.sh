#!/bin/bash

# Fixed values for the experiment
scale="m"
n_workers="10"


# Array of methods "NSGA2" is not working currencly
methods=("RS" "RSBO" "LSBO" "NSGA2")
seeds=("31415927" "42" "6790")
devices=("rtx2080" "h100" "cpu_xeon_gold" "cpu_amd_7452")
objectives=("energies" "latencies") # perplexity is always part of the evaluations


for method in "${methods[@]}"
do  
    for seed in "${seeds[@]}"
    do
        for device in "${devices[@]}"
        do
            for objective in "${objectives[@]}"
            do
                echo "Evaluating $method with seed $seed on $device for $objective"
                python run_nas_gpt_2d.py \
                    --method "$method" \
                    --metric "$objective" \
                    --n_workers "$n_workers" \
                    --random_seed "$seed" \
                    --device "$device" \
                    --search_space "$scale" \
                    --objective "$objective"
            done
        done
    done
done

