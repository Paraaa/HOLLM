from mooLLM.llm.models.local import LOCAL
from mooLLM.llm.models.gpt import GPT
from mooLLM.llm.models.gemini import GEMINI
from mooLLM.llm.models.groq import GROQ
from mooLLM.llm.models.huggingface import HUGGINGFACE
from mooLLM.llm.models.deepseek import DEEPSEEK

from mooLLM.space_partitioning.voronoi_partitioning import VoronoiPartitioning
from mooLLM.space_partitioning.kd_tree_partitioning import KDTreePartitioning


from mooLLM.acquisition_functions.function_value import FunctionValueACQ

from mooLLM.region_acquisition_functions.score_region_acq import ScoreRegionACQ
from mooLLM.surrogate_models.llm_surrogate_batch import LLM_Surrogate_batch
from mooLLM.surrogate_models.llm_surrogate import LLM_Surrogate
from mooLLM.candidate_sampler.LLM_sampler import LLM_SAMPLER
from mooLLM.warmstarter.random_warmstarter import RANDOM_WARMSTARTER
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
    "gpt-4o-mini": GPT("gpt-4o-mini"),
    "gemini-1-5-flash": GEMINI("gemini-1.5-flash"),
}

WARMSTARTERS = {
    "RANDOM_WARMSTARTER": RANDOM_WARMSTARTER(),
}
CANDIDATE_SAMPLERS = {
    "LLM_SAMPLER": LLM_SAMPLER(),
}
SURROGATE_MODELS = {
    "LLM_SUR": LLM_Surrogate(),
    "LLM_SUR_BATCH": LLM_Surrogate_batch(),
}
ACQUISITION_FUNCTIONS = {
    "FunctionValueACQ": FunctionValueACQ(),  # Single objective
}

REGION_ACQUISITION_FUNCTIONS = {
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

SPACE_PARTITIONING_STRATEGIES = {
    "voronoi": VoronoiPartitioning(),
    "kdtree": KDTreePartitioning(),
}
