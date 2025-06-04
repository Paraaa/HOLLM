from mooLLM.llm.models.local import LOCAL
from mooLLM.llm.models.gpt import GPT
from mooLLM.llm.models.dummy_nb201 import DUMMY_NB201
from mooLLM.llm.models.dummy_hwgpt import DUMMY_HWGPT
from mooLLM.llm.models.gemini import GEMINI
from mooLLM.llm.models.groq import GROQ
from mooLLM.llm.models.huggingface import HUGGINGFACE
from mooLLM.llm.models.deepseek import DEEPSEEK

from mooLLM.space_partitioning.voronoi_partitioning import VoronoiPartitioning
from mooLLM.space_partitioning.kd_tree_partitioning import KDTreePartitioning

from mooLLM.acquisition_functions.hypervolume_improvement import HypervolumeImprovement
from mooLLM.acquisition_functions.hypervolume_improvement_batch import (
    HypervolumeImprovementBatch,
)

from mooLLM.acquisition_functions.random_acq import RandomACQ
from mooLLM.acquisition_functions.function_value import FunctionValueACQ
from mooLLM.region_acquisition_functions.volume_region_acq import VolumeRegionACQ
from mooLLM.region_acquisition_functions.vis_he_region_acq import VISHERegionACQ
from mooLLM.region_acquisition_functions.vis_he_scheduling_region_acq import (
    VISHESchedulingRegionACQ,
)
from mooLLM.region_acquisition_functions.vis_region_acq import VISRegionACQ
from mooLLM.region_acquisition_functions.score_region_acq import ScoreRegionACQ
from mooLLM.surrogate_models.llm_surrogate_batch import LLM_Surrogate_batch
from mooLLM.surrogate_models.llm_surrogate import LLM_Surrogate
from mooLLM.candidate_sampler.LLM_sampler import LLM_SAMPLER
from mooLLM.candidate_sampler.tot_sampler import TreeOfThoughtSampler
from mooLLM.warmstarter.random_warmstarter import RANDOM_WARMSTARTER
from mooLLM.warmstarter.zero_shot_warmstarter import ZERO_SHOT_WARMSTARTER
from mooLLM.warmstarter.checkpoint_warmstarter import CHECKPOINT_WARMSTARTER
from mooLLM.statistics.context_limit_strategy.lastN import LastN
from mooLLM.statistics.context_limit_strategy.random import Random
from mooLLM.schedulers.constant_scheduler import ConstantScheduler
from mooLLM.schedulers.step_wise_decay_scheduler import StepWiseDecayScheduler
from mooLLM.schedulers.linear_decay_scheduler import LinearDecayScheduler
from mooLLM.schedulers.epsilon_greedy_scheduler import EpsilonGreedyScheduler
from mooLLM.schedulers.epsilon_greedy_decay_scheduler import EpsilonDecayScheduler
from mooLLM.schedulers.cosine_annealing_scheduler import CosineAnnealingScheduler
from mooLLM.schedulers.cosine_decay_scheduler import CosineDecayScheduler


MODELS = {
    "llama-3-8b-instruct-awq": LOCAL("casperhansen/llama-3-8b-instruct-awq"),
    "gemma-2-9b-it-AWQ": LOCAL("solidrust/gemma-2-9b-it-AWQ"),
    "Qwen2.5-7B-Instruct-AWQ": LOCAL("Qwen/Qwen2.5-7B-Instruct-AWQ"),
    "Qwen2.5-14B-Instruct-AWQ": LOCAL("Qwen/Qwen2.5-14B-Instruct-AWQ"),
    "Qwen2.5-32B-Instruct-AWQ": LOCAL("Qwen/Qwen2.5-32B-Instruct-AWQ"),
    "QwQ-32B-AWQ": LOCAL("Qwen/QwQ-32B-AWQ"),
    "phi-4-AWQ": LOCAL("stelterlab/phi-4-AWQ"),
    "gpt-4o-mini": GPT("gpt-4o-mini"),
    "gpt-4o-mini-space-partitioning": GPT("gpt-4o-mini"),
    "gemini-1-5-flash": GEMINI("gemini-1.5-flash"),
    "deepseek-qwen-14B-base": DEEPSEEK("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
    # "QwQ-32B-AWQ": DEEPSEEK("Qwen/QwQ-32B-AWQ"),
    "dummy_nb201": DUMMY_NB201("Dummy-nb201"),
    "dummy_hwgpt": DUMMY_HWGPT("Dummy-hwgpt"),
}

WARMSTARTERS = {
    "RANDOM_WARMSTARTER": RANDOM_WARMSTARTER(),
    "ZERO_SHOT_WARMSTARTER": ZERO_SHOT_WARMSTARTER(),
    "CHECKPOINT_WARMSTARTER": CHECKPOINT_WARMSTARTER(),
}
CANDIDATE_SAMPLERS = {
    "LLM_SAMPLER": LLM_SAMPLER(),
    "TOT_SAMPLER": TreeOfThoughtSampler(),
}
SURROGATE_MODELS = {
    "LLM_SUR": LLM_Surrogate(),
    "LLM_SUR_BATCH": LLM_Surrogate_batch(),
}
ACQUISITION_FUNCTIONS = {
    "HypervolumeImprovement": HypervolumeImprovement(),  # Multi-objective
    "HypervolumeImprovementBatch": HypervolumeImprovementBatch(),  # Multi-objective, Deprecated only used for ToT. Please use the non-batch version
    "RandomACQ": RandomACQ(),  # Single and multi-objective
    "FunctionValueACQ": FunctionValueACQ(),  # Single objective
}

REGION_ACQUISITION_FUNCTIONS = {
    "VOLUME": VolumeRegionACQ(),
    "VIS_HE": VISHERegionACQ(),
    "VIS_HE_SCHEDULER": VISHESchedulingRegionACQ(),
    "VIS": VISRegionACQ(),
    "ScoreRegion": ScoreRegionACQ(),
}

SCHEDULERS = {
    "CONSTANT_SCHEDULER": ConstantScheduler(),
    "STEP_WISE_DECAY_SCHEDULER": StepWiseDecayScheduler(),
    "LINEAR_DECAY_SCHEDULER": LinearDecayScheduler(),
    "EPSILON_GREEDY_SCHEDULER": EpsilonGreedyScheduler(),
    "EPSILON_DECAY_SCHEDULER": EpsilonDecayScheduler(),
    "COSINE_ANNEALING_SCHEDULER": CosineAnnealingScheduler(),
    "COSINE_DECAY_SCHEDULER": CosineDecayScheduler(),
}

CONTEXT_LIMIT_STRATEGIES = {
    "LastN": LastN(),
    "Random": Random(),
}
# Space partitioning strategies
SPACE_PARTITIONING_STRATEGIES = {
    "voronoi": VoronoiPartitioning(),
    "kdtree": KDTreePartitioning(),
}
