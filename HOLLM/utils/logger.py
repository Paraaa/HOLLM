import os
import logging
from datetime import datetime

timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")


LOGDIR = ".mooLLM/log/"
LOGFILE_TEMPLATE = "mooLLM_{}.log"
LOGFILE = LOGFILE_TEMPLATE.format(timestamp)
LOGPATH = os.path.join(LOGDIR, LOGFILE)

os.makedirs(LOGDIR, exist_ok=True)


LOGGING_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s"
DEFAULT_LOG_LEVEL = 0  # 0 equals to NOTSET level

# Estimate maxBytes based on average log entry size (assume 100 bytes per entry)
MAX_BYTES = 10000 * 100  # ~1 MB assuming each entry is about 100 bytes

# Add a the logger name to this list to be shown in the output logs and console
WHITELISTED_LOGGERS = [
    "mooLLM",
    "mooLLM",
    "LLMInterface",
    "Builder",
    "ACQUISITION_FUNCTION",
    "LOCAL_LLM",
    "Dummy_LLM",
    "mooLLM_ACQ",
    "RateLimiter",
    "Hypervolume",
    "SURROGATE_MODEL",
    "CANDIDATE_SAMPLER",
    "LLM_SAMPLER",
    "WARMSTARTER",
    "RANDOM_WARMSTARTER",
    "ZERO_SHOT_WARMSTARTER",
    "PromptBuilder",
    "STATISTICS",
    "HW_GPT_BENCH",
    "RandomACQ",
    "WHITTLE",
    "KDE",
    "GPT",
    "ZDT",
    "BENCHMARK",
    "TREE_OF_THOUGHT_SAMPLER",
    "SURROGATE_MODEL_BATCH",
    "HypervolumeImprovementBatch",
    "mooLLM_ToT",
    "CHECKPOINT_WARMSTARTER",
    "mooLLMInterval",
    "SpacePartitioningMOOLLM",
    "RegionACQ",
    "ThreadedMOOLLM",
    "VoronoiPartitioning",
    "WELDED_BEAM",
    "ZINC256",
    "KDTreePartitioning",
    "GEMINI",
    "HUGGINGFACE",
    "Groq",
    "COSINE_ANNEALING_SCHEDULER",
    "LLM_Surrogate",
    "LLM_Surrogate_batch",
    "MOSSRegionACQ",
]


class WhitelistFilter(logging.Filter):
    def __init__(self, whitelist=None):
        super().__init__()
        if whitelist is None:
            whitelist = []
        self.whitelist = whitelist

    def filter(self, record):
        return record.name in self.whitelist


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": LOGGING_FORMAT}},
    "filters": {
        "whitelist_filter": {
            "()": WhitelistFilter,
            "whitelist": WHITELISTED_LOGGERS,
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
            "filters": ["whitelist_filter"],
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "simple",
            "filename": LOGPATH,
            "filters": ["whitelist_filter"],
        },
    },
    "loggers": {"root": {"level": "DEBUG", "handlers": ["stdout", "file"]}},
}
