#!/bin/bash
seeds=("31415927" "42" "6790")
model="Qwen2.5-32B-Instruct-AWQ"


config_file="NB201/mooLLM"
method_name="mooLLM (Qwen2.5-32B)"
for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done

config_file="NB201/mooLLM-VP"
method_name="mooLLM-VP (Qwen2.5-32B)"
for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done


config_file="NB201/mooLLM-VP-VIS"
method_name="mooLLM-VP-VIS (Qwen2.5-32B)"
for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done


config_file="NB201/mooLLM-VP-PE"
method_name="mooLLM-VP-PE (Qwen2.5-32B)"
for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done