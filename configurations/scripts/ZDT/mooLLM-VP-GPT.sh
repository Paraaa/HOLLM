#!/bin/bash
config_file="ZDT/mooLLM-VP-VIS-HE-ALL"
seeds=("31415927" "42" "6790")

model="gpt-4o-mini"
method_name="mooLLM-VP-VIS-HE-ALL-SCHEDULING (GPT-4o Mini)"

for seed in "${seeds[@]}"
do
    echo "Evaluating ZDT with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed"\
        --model="$model" \
        --method_name="$method_name"
done