# HOLLM: Improving LLM-based Global Optimization with Search Space Partitioning
<p align="center">
  <img src="./logo.png" alt="HOLLM Logo" width="150" height="150">
</p>

This repository contains the official implementation of [HOLLM](https://arxiv.org/abs/2505.21372v1) *(Hierarchical Optimization with Large Language Models)*.

## Installation (Python 3.11)

1.  Create a conda environment:
    ```bash
    conda create -n HOLLM python=3.11
    conda activate HOLLM
    ```

2.  Install package and install dependencies:
    ```bash
    pip install -e .
    pip install -r requirements.txt
    ```

3. Install [SyneTune](https://github.com/syne-tune/syne-tune) locally and switch to the `llmkd` branch. Note that you also need to install the following dependencies from SyneTune:
    ```bash
    pip install -r requirements-bore.txt
    pip install -r requirements-botorch.txt
    pip install -r requirements-kde.txt
    pip install -r requirements-smac.txt
    ```

4.  Set the required API keys as environment variables in your shell configuration (for example, in your `~/.bashrc` or `~/.zshrc`):

    ```bash
    export OPEN_AI_API_KEY="<YOUR_OPEN_AI_API_KEY>"
    export GOOGLE_AI_API_KEY="<YOUR_GOOGLE_AI_API_KEY>"
    ```

    Reload your terminal or source the configuration file to apply the changes. If you don't have access to certain APIs, you can set the corresponding keys to an empty string.

## How to run evaluations from the paper

The evaluations and experiments presented in the accompanying paper can be reproduced using the scripts located in the `experiments/` directory.

-   **Synthetic Benchmarks:** To run experiments on synthetic benchmark functions, use the `run_synthetic.sh` script.
    ```bash
    ./experiments/run_synthetic.sh
    ```
-   **FCNet Benchmarks:** To run experiments on the FCNet tabular benchmark, use the `run_fcnet.sh` script.
    ```bash
    ./experiments/run_fcnet.sh
    ```
-   **NB201 Benchmarks:** For NB201 we opted for a different prompting strategy. Implementation details can be found in this repository: [HOLLM](https://github.com/automl/hollm/tree/main )

-   **Ablation Studies:** To run ablation studies, use the `run_ablations.sh` script.
    ```bash
    ./experiments/run_ablations.sh
    ```

Please inspect the shell scripts for details on the specific configurations and benchmarks being run. Results are typically saved in the `results/` directory.


## How to use HOLLM
HOLLM can be integrated into hyperparameter optimization workflows. The core component is the `mooLLM` class, which is configured and instantiated using a `Builder`. The package can be installed locally using:
   ```bash
   pip install -e .
   ```

Here is a simple example on how to run:

```python
from HOLLM.builder import Builder

# To optimize a custom black box function you need to implement the `BENCHMARK` base class. Refer to the abstract base class for more details or the SyneTuneBenchmark for more details.
from HOLLM.benchmarks.benchmark import BENCHMARK

class YourBenchmark(BENCHMARK):
    def __init__(self):
        super().__init__()

    def generate_initialization(self, n_points: int, **kwargs):
        # Return list of n_points configuration dictionaries
        pass

    def evaluate_point(self, point, **kwargs) -> Dict:
        # Return evaluation metrics for the given point
        pass

    def get_few_shot_samples(self, **kwargs):
        # Return list of (config, evaluation) tuples for prompting
        pass

    def get_metrics_ranges(self, **kwargs):
        # Return dict mapping metric names to their [min, max] ranges
        pass

    def is_valid_candidate(self, candidate: Dict) -> bool:
        # Validate candidate configuration
        pass

    def is_valid_evaluation(self, evaluation: Dict) -> bool:
        # Validate evaluation results
        pass
```



```python
# You need to define a config for the builder to build the instance of the optimization strategy.
# **Refer** to the builder class and the synetune implementation in synetune_utils.py for more details.
config = {...}
builder: Builder = Builder(config=config, benchmark=YourBenchmark())
instance = builder.build() # This returns a optimization_strategy instance
instance.optimize() # Runs the optimization loop
```

A list of available setting for the config can be listed with:
```bash
python main.py --help
```
or in the `Builder(...)` implementation.

## License

This project is licensed under the terms of the MIT License. See the `LICENSE` file for more details.
