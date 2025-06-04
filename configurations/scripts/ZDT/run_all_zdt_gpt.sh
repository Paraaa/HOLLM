#!/bin/bash

model="gpt-4o-mini"
method_name="mooLLM (GPT-4o Mini)"
seeds=("31415927" "42" "6790")

# ZDT1 
config_file="ZDT1/mooLLM"
for seed in "${seeds[@]}"
do
    echo "Evaluating ZDT1 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed"\
        --model="$model" \
        --method_name="$method_name"
done

# ZDT2 
config_file="ZDT2/mooLLM"
for seed in "${seeds[@]}"
do
    echo "Evaluating ZDT2 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed"\
        --model="$model" \
        --method_name="$method_name"
done

# # ZDT3
# config_file="ZDT3/mooLLM"
# for seed in "${seeds[@]}"
# do
#     echo "Evaluating ZDT3 with seed $seed"
#     python main.py \
#         --config_file="$config_file" \
#         --seed="$seed"\
#         --model="$model" \
#         --method_name="$method_name"
# done

# ZDT4
config_file="ZDT4/mooLLM"
for seed in "${seeds[@]}"
do
    echo "Evaluating ZDT4 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed"\
        --model="$model" \
        --method_name="$method_name"
done

# ZDT6
config_file="ZDT6/mooLLM"
for seed in "${seeds[@]}"
do
    echo "Evaluating ZDT6 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed"\
        --model="$model" \
        --method_name="$method_name"
done
