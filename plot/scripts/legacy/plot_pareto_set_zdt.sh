#!/bin/bash
# python plot_pareto_set_zdt.py --data_path="./ablations_zdt/ZDT" --save_dir_name="baselines" --seed=42
# python plot_pareto_set_zdt.py --data_path="./ablations_zdt/ZDT" --save_dir_name="baselines" --seed=6790
# python plot_pareto_set_zdt.py --data_path="./ablations_zdt/ZDT" --save_dir_name="baselines" --seed=31415927
python plot_pareto_set_zdt.py --data_path="./ablations_zdt/ZDT_ToT" --save_dir_name="baseline/ToT" --seed=42
python plot_pareto_set_zdt.py --data_path="./ablations_zdt/ZDT_ToT" --save_dir_name="baseline/ToT" --seed=6790
python plot_pareto_set_zdt.py --data_path="./ablations_zdt/ZDT_ToT" --save_dir_name="baseline/ToT" --seed=31415927
python plot_pareto_set_zdt.py --data_path="./ablations_zdt/ZDT_ToT_cherry" --save_dir_name="baseline/ToT_cherry" --seed=42