benchmarks=("penicillin" "vehicle_safety" "car_side_impact")
model="gemini-2.0-flash"

# # Run the vanilla method for each benchmark
for benchmark in "${benchmarks[@]}"; do
    python experiments/benchmark_main_synthetic.py \
        --max_num_evaluations 100 \
        --seed 3 \
        --run_all_seeds 1 \
        --method LLMKD \
        --method_name "LLM" \
        --benchmark "$benchmark" \
        --model $model \
        --optimization_method "mooLLM" \
        --candidates_per_request 20 \
        --n_workers 1
done

# Run the LLMKD method for each benchmark
for benchmark in "${benchmarks[@]}"; do
    python experiments/benchmark_main_synthetic.py \
        --max_num_evaluations 100 \
        --seed 3 \
        --run_all_seeds 1 \
        --method LLMKD \
        --method_name "LLMKD" \
        --benchmark "$benchmark" \
        --model $model \
        --optimization_method "SpacePartitioning" \
        --m0 0.5 \
        --lam 0 \
        --candidates_per_request 5 \
        --partitions_per_trial 5 \
        --alpha_max 1.0 \
        --n_workers 1 \
        --use_dimension_scaling 1
done

methods=("RS" "BORE" "TPE" "CQR" "LLMKD" "BOTorch" "REA")
for method in "${methods[@]}"; do
    # Run the baseline methods for each benchmark
    for benchmark in "${benchmarks[@]}"; do
        python experiments/benchmark_main_synthetic.py \
            --max_num_evaluations 100 \
            --seed 3 \
            --run_all_seeds 1 \
            --method $method \
            --benchmark "$benchmark"
    done
done
