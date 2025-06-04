script="plot_fvals_paper.py"

blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,alpha-0.2,alpha-0.5,alpha-0.7"


# benchmarks=("nas201-cifar10" "nas201-cifar100" "nas201-ImageNet16-120")
# benchmarks=("fcnet-naval" "fcnet-parkinsons" "fcnet-protein" "fcnet-slice" "tabrepo-CatBoost-2dplanes" "tabrepo-CatBoost-Airlines-DepDelay-10M" "tabrepo-CatBoost-Allstate-Claims-Severity" "tabrepo-CatBoost-Amazon-employee-access" "tabrepo-CatBoost-APSFailure" "tabrepo-CatBoost-Australian" "tabrepo-CatBoost-Bioresponse" "tabrepo-CatBoost-Brazilian-houses" "tabrepo-CatBoost-Buzzinsocialmedia-Twitter" "tabrepo-CatBoost-CIFAR-10" "tabrepo-RandomForest-2dplanes" "tabrepo-RandomForest-Airlines-DepDelay-10M" "tabrepo-RandomForest-Allstate-Claims-Severity" "tabrepo-RandomForest-Amazon-employee-access" "tabrepo-RandomForest-APSFailure" "tabrepo-RandomForest-Australian" "tabrepo-RandomForest-Bioresponse" "tabrepo-RandomForest-Brazilian-houses" "tabrepo-RandomForest-Buzzinsocialmedia-Twitter" "tabrepo-RandomForest-CIFAR-10")
# for benchmark in "${benchmarks[@]}"g
# do
#     python ./plot/$script \
#         --benchmark $benchmark \
#         --title "Fvals ($benchmark)" \
#         --columns "F1" \
#         --trials 100 \
#         --data_path ./results/SyneTune/$benchmark/ \
#         --filename "fvals_$benchmark" \
#         --normalization_method "none" \
#         --blacklist $blacklist \
#         --use_log_scale "False"
# done

# --- THE SIX FUNCTIONS ---
#python ./plot/$script \
#        --benchmark "ackley"  \
#        --title "Ackley" \
#        --columns "F1" \
#        --trials 100 \
#        --data_path ./results/SyneTune/ackley/ \
#        --filename "fvals_ackley" \
#        --normalization_method "none" \
#        --blacklist $blacklist \
#        --use_log_scale "False"
#
#python ./plot/$script \
#        --benchmark "hartmann3"  \
#        --title "Hartmann3" \
#        --columns "F1" \
#        --trials 100 \
#        --data_path ./results/SyneTune/hartmann3/ \
#        --filename "fvals_hartmann3" \
#        --normalization_method "none" \
#        --blacklist $blacklist \
#        --use_log_scale "False"
#
#python ./plot/$script \
#        --benchmark "hartmann6"  \
#        --title "Hartmann6" \
#        --columns "F1" \
#        --trials 100 \
#        --data_path ./results/SyneTune/hartmann6/ \
#        --filename "fvals_hartmann6" \
#        --normalization_method "none" \
#        --blacklist $blacklist \
#        --use_log_scale "False"


blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,alpha-0.2,alpha-0.5,alpha-0.7,alpha-1.0,LLMKD-k-1,LLMKD-k-3,LLMKD-k-5,LLMKD-k-7,LLMKD-k-10,LLMKD-m0-0.5*d,LLMKD-m0-1*d,LLMKD-m0-0.25*d,LLMKD-m0-1"
python ./plot/$script \
        --benchmark "levy"  \
        --title "Levy (Number of Partitions)" \
        --columns "F1" \
        --trials 100 \
        --data_path ./results/SyneTune/levy/ \
        --filename "fvals_levy_num_partitions" \
        --normalization_method "none" \
        --blacklist $blacklist \
        --use_log_scale "False" \
        --y_lim "-30" "0"

blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,alpha-0.2,alpha-0.5,alpha-0.7,LLMKD-k-1,LLMKD-k-3,LLMKD-k-5,LLMKD-k-7,LLMKD-k-10,LLMKD-1-partition-per-trial,LLMKD-3-partition-per-trial,LLMKD-5-partition-per-trial,LLMKD-7-partition-per-trial"
python ./plot/$script \
        --benchmark "levy"  \
        --title "Levy (Leaf Size)" \
        --columns "F1" \
        --trials 100 \
        --data_path ./results/SyneTune/levy/ \
        --filename "fvals_levy_m0" \
        --normalization_method "none" \
        --blacklist $blacklist \
        --use_log_scale "False" \
        --y_lim "-30" "0"


blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,alpha-0.2,alpha-0.5,alpha-0.7,LLMKD-m0-0.5*d,LLMKD-m0-1*d,LLMKD-m0-0.25*d,LLMKD-m0-1,LLMKD-1-partition-per-trial,LLMKD-3-partition-per-trial,LLMKD-5-partition-per-trial,LLMKD-7-partition-per-trial"
python ./plot/$script \
        --benchmark "levy"  \
        --title "Levy (Number of candidates per region)" \
        --columns "F1" \
        --trials 100 \
        --data_path ./results/SyneTune/levy/ \
        --filename "fvals_levy_candidates_per_region" \
        --normalization_method "none" \
        --blacklist $blacklist \
        --use_log_scale "False" \
        --y_lim "-30" "0"



#python ./plot/$script \
#        --benchmark "rosenbrock"  \
#        --title "Rosenbrock" \
#        --columns "F1" \
#        --trials 100 \
#        --data_path ./results/SyneTune/rosenbrock/ \
#        --filename "fvals_rosenbrock" \
#        --normalization_method "none" \
#        --blacklist $blacklist \
#        --use_log_scale "False" \
#        --y_lim "-2000" "20"
#
#
#python ./plot/$script \
#        --benchmark "rastrigin"  \
#        --title "Rastrigin" \
#        --columns "F1" \
#        --trials 100 \
#        --data_path ./results/SyneTune/rastrigin/ \
#        --filename "fvals_rastrigin" \
#        --normalization_method "none" \
#        --blacklist $blacklist \
#        --use_log_scale "False"
## # --- THE SIX FUNCTIONS END ---
#
## # --- FCNET ---
#python ./plot/$script \
#        --benchmark "fcnet-naval" \
#        --title "Naval" \
#        --columns "F1" \
#        --trials 100 \
#        --data_path ./results/SyneTune/fcnet-naval/ \
#        --filename "fvals_fcnet-naval" \
#        --normalization_method "none" \
#        --blacklist $blacklist \
#        --use_log_scale "False" \
#        --y_lim "-0.0004" "0"
#
#
#python ./plot/$script \
#        --benchmark "fcnet-parkinsons" \
#        --title "Parkinsons" \
#        --columns "F1" \
#        --trials 100 \
#        --data_path ./results/SyneTune/fcnet-parkinsons/ \
#        --filename "fvals_fcnet-parkinsons" \
#        --normalization_method "none" \
#        --blacklist $blacklist \
#        --use_log_scale "False" \
#        --y_lim "-0.035" "-0.005"
#
#
#python ./plot/$script \
#        --benchmark "fcnet-protein" \
#        --title "Protein" \
#        --columns "F1" \
#        --trials 100 \
#        --data_path ./results/SyneTune/fcnet-protein/ \
#        --filename "fvals_fcnet-protein" \
#        --normalization_method "none" \
#        --blacklist $blacklist \
#        --use_log_scale "False" \
#        --y_lim "-0.30" "-0.22"
#
#python ./plot/$script \
#        --benchmark "fcnet-slice" \
#        --title "Slice" \
#        --columns "F1" \
#        --trials 100 \
#        --data_path ./results/SyneTune/fcnet-slice/ \
#        --filename "fvals_fcnet-slice" \
#        --normalization_method "none" \
#        --blacklist $blacklist \
#        --use_log_scale "False" \
#        --y_lim "-0.0010" "-0.0001"
#
## --- FCNET END---
#
## --- FCNET Alpha comparision ---

#blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,BORE,BOTorch,CQR,RS,TPE,RE"
blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,BORE,BOTorch,CQR,RS,TPE,RE,LLM-25,LLMKD-k-1,LLMKD-k-3,LLMKD-k-5,LLMKD-k-7,LLMKD-k-10,LLMKD-m0-0.5*d,LLMKD-m0-1*d,LLMKD-m0-0.25*d,LLMKD-m0-1,LLMKD-1-partition-per-trial,LLMKD-3-partition-per-trial,LLMKD-5-partition-per-trial,LLMKD-7-partition-per-trial"
python ./plot/$script \
        --benchmark "fcnet-naval" \
        --title "Naval" \
        --columns "F1" \
        --trials 100 \
        --data_path ./results/SyneTune/fcnet-naval/ \
        --filename "alpha-comparision/fvals_fcnet-naval" \
        --normalization_method "none" \
        --blacklist $blacklist \
        --use_log_scale "False" \
        --y_lim "-0.00010" "-0.00002"

blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,BORE,BOTorch,CQR,RS,TPE,RE,LLM-25,LLMKD-k-1,LLMKD-k-3,LLMKD-k-5,LLMKD-k-7,LLMKD-k-10,LLMKD-m0-0.5*d,LLMKD-m0-1*d,LLMKD-m0-0.25*d,LLMKD-m0-1,LLMKD-1-partition-per-trial,LLMKD-3-partition-per-trial,LLMKD-5-partition-per-trial,LLMKD-7-partition-per-trial"
python ./plot/$script \
        --benchmark "fcnet-parkinsons" \
        --title "Parkinsons" \
        --columns "F1" \
        --trials 100 \
        --data_path ./results/SyneTune/fcnet-parkinsons/ \
        --filename "alpha-comparision/fvals_fcnet-parkinsons" \
        --normalization_method "none" \
        --blacklist $blacklist \
        --use_log_scale "False" \
        --y_lim "-0.025" "-0.010"

blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,BORE,BOTorch,CQR,RS,TPE,RE,LLM-25,LLMKD-k-1,LLMKD-k-3,LLMKD-k-5,LLMKD-k-7,LLMKD-k-10,LLMKD-m0-0.5*d,LLMKD-m0-1*d,LLMKD-m0-0.25*d,LLMKD-m0-1,LLMKD-1-partition-per-trial,LLMKD-3-partition-per-trial,LLMKD-5-partition-per-trial,LLMKD-7-partition-per-trial"
python ./plot/$script \
        --benchmark "fcnet-protein" \
        --title "Protein" \
        --columns "F1" \
        --trials 100 \
        --data_path ./results/SyneTune/fcnet-protein/ \
        --filename "alpha-comparision/fvals_fcnet-protein" \
        --normalization_method "none" \
        --blacklist $blacklist \
        --use_log_scale "False" \
        --y_lim "-0.26" "-0.225"
blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,BORE,BOTorch,CQR,RS,TPE,RE,LLM-25,LLMKD-k-1,LLMKD-k-3,LLMKD-k-5,LLMKD-k-7,LLMKD-k-10,LLMKD-m0-0.5*d,LLMKD-m0-1*d,LLMKD-m0-0.25*d,LLMKD-m0-1,LLMKD-1-partition-per-trial,LLMKD-3-partition-per-trial,LLMKD-5-partition-per-trial,LLMKD-7-partition-per-trial"
python ./plot/$script \
        --benchmark "fcnet-slice" \
        --title "Slice" \
        --columns "F1" \
        --trials 100 \
        --data_path ./results/SyneTune/fcnet-slice/ \
        --filename "alpha-comparision/fvals_fcnet-slice" \
        --normalization_method "none" \
        --blacklist $blacklist \
        --use_log_scale "False" \
        --y_lim "-0.0006" "-0.0002"
#
## --- FCNET Alpha comparision END ---