import logging
from mooLLM.benchmarks.nb201.NB201Bench import NB201
from mooLLM.benchmarks.hw_gpt_bench.hw_gpt_bench import HWGPTBENCH
from mooLLM.benchmarks.welded_beam.welded_beam import WELDED_BEAM
from mooLLM.benchmarks.zinc256.zinc256 import ZINC256
from mooLLM.benchmarks.dtlz.dtlz import DTLZ
from mooLLM.benchmarks.vlmop.vlmop import VLMOP

# from mooLLM.benchmarks.whittle.whittle import WHITTLE
from mooLLM.benchmarks.zdt.zdt import ZDT

logger = logging.getLogger("Benchmark Initialization")


def get_benchmark_fn(config):
    benchmark_settings = config.get("benchmark_settings", None)
    benchmark_name = config.get("benchmark", None)
    seed = config.get("seed", None)
    metrics = config.get("metrics", None)
    model_name = config.get("llm_settings", {}).get("model", None)

    match benchmark_name:
        case "NB201":
            dataset = benchmark_settings.get("dataset", None)
            device_metric = benchmark_settings.get("device_metric", None)
            if dataset is None:
                logger.warning("No dataset specified for NB201 benchmark")
                raise ValueError("No dataset specified for NB201 benchmark")
            if device_metric is None:
                logger.warning("No device metric specified for NB201 benchmark")
                raise ValueError("No device metric specified for NB201 benchmark")

            return NB201(metrics, dataset, device_metric, seed, model_name)
        case "HWGPT":
            scale = benchmark_settings.get("scale", None)
            device = benchmark_settings.get("device", None)
            predictor = benchmark_settings.get("predictor", None)
            use_supernet_surrogate = benchmark_settings.get(
                "use_supernet_surrogate", None
            )
            return HWGPTBENCH(
                scale,
                use_supernet_surrogate,
                metrics,
                device,
                predictor,
                model_name,
                seed,
            )
        case "ZDT":
            problem_id = benchmark_settings.get("problem_id", 1)
            return ZDT(model_name, seed, problem_id)
        case "WELDED_BEAM":
            problem_id = benchmark_settings.get("problem_id", None)
            return WELDED_BEAM(model_name, seed, problem_id)
        case "ZINC256":
            return ZINC256(model_name, seed)
        case "VLMOP":
            return VLMOP(model_name, seed)
        case "DTLZ":
            problem_id = benchmark_settings.get("problem_id", 1)
            return DTLZ(model_name, seed)
        case _:
            raise ValueError(f"Unsupported benchmark: {benchmark_name}")
