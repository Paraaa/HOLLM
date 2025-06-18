from typing import List, Dict
from HOLLM.space_partitioning.utils import Region, BoundingBox
from abc import ABC, abstractmethod
from HOLLM.statistics.statistics import Statistics


class SPACE_PARTITIONING_STRATEGY(ABC):
    def __init__(self):
        self.bounding_box: BoundingBox = None
        self.statistics: Statistics = None
        self.space_partitioning_settings: dict = None

    @abstractmethod
    def partition(self, points: List[Dict], fvals: List[Dict]) -> List[Region]:
        pass
