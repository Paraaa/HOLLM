# Note this doesnt inhereit from ACQUISITION_FUNCTION as it is only applied to region selection
import logging
from typing import List
from abc import ABC, abstractmethod
from mooLLM.space_partitioning.utils import Region
from mooLLM.schedulers.scheduler import Scheduler

logger = logging.getLogger("RegionACQ")


class RegionACQ(ABC):
    """
    Abstract base class for region acquisition functions. Inherit from this class to implement specific strategies.
    """

    def __init__(self):
        self.strategy: str = None
        self.alpha: float = 0.5
        self.scheduler: Scheduler = None
        self.n_trials: int = None
        self.metrics_targets: List[str] = None
        self.space_partitioning_settings: dict = None
        logger.debug("Initialized RegionACQ base class.")

    @abstractmethod
    def select_regions(self, regions: List[Region], num_regions: int):
        pass
