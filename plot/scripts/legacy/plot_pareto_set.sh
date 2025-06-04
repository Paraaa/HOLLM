#!/bin/bash
python plot_pareto_set.py --data_path="./ablations_hwgpt/poster/" --save_dir_name="baseline" --seed=42
python plot_pareto_set.py --data_path="./ablations_hwgpt/poster/" --save_dir_name="baseline" --seed=6790
python plot_pareto_set.py --data_path="./ablations_hwgpt/poster/" --save_dir_name="baseline" --seed=31415927
