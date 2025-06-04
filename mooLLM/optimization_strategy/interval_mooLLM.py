import logging
from typing import Dict
from mooLLM.optimization_strategy.optimization_strategy import OptimizationStrategy
from mooLLM.optimization_strategy.mooLLM import mooLLM
from mooLLM.optimization_strategy.tot_mooLLM import mooLLMToT
from mooLLM.statistics.statistics import Statistics

logger = logging.getLogger("mooLLMInterval")


class mooLLMInterval(OptimizationStrategy):
    def __init__(self):
        self.mooLLM: mooLLM = None
        self.mooLLM_tot: mooLLMToT = None
        self.interval_settings: Dict = None
        self.n_trials: int = None
        self.statistics: Statistics = None

    def initialize(self):
        pass

    # TODO: This works but the trials are not correct yet. Maybe I really need to pass them here
    def optimize(self) -> Dict:
        logger.debug("Starting mooLLMInterval optimization.")
        for trial in range(self.n_trials):
            logger.debug(f"Trial {self.mooLLM.current_trial}")
            self.mooLLM.optimize()
            self.mooLLM_tot.optimize()
        logger.info("mooLLMInterval optimization completed.")
        return self.statistics.get_statistics()
