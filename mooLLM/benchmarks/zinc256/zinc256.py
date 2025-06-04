import os
import logging
import pandas as pd
from mooLLM.benchmarks.benchmark import BENCHMARK

logger = logging.getLogger("ZINC256")


class ZINC256(BENCHMARK):
    def __init__(self, model_name, seed):
        super().__init__()
        self.name = "ZINC256"
        self.dataset = "zinc256"
        self.metrics = ["logP", "qed"]
        self.model_name = model_name
        self.seed = seed

        self.dataset = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "250k_rndm_zinc_drugs_clean_3.csv",
            )
        )

    def generate_initialization(self, n_points, **kwargs):
        points = [
            {"smiles": record["smiles"]}
            for record in self.dataset.sample(n=n_points).to_dict(orient="records")
        ]
        logger.debug(f"Generated initialization points: {points}")
        return points

    def evaluate_point(self, point, **kwargs):
        logger.debug(f"Evaluating point: {point}")
        # 'point' is a SMILES representation. Find its matching record in the dataset.
        match = self.dataset[self.dataset["smiles"] == f"{point['smiles']}"]
        if match.empty:
            raise ValueError(f"SMILES '{point}' not found in dataset.")
        record = match.iloc[0]
        return point, {"logP": round(record["logP"], 3), "qed": round(record["qed"], 3)}

    def get_few_shot_samples(self, **kwargs):
        return

    def get_metrics_ranges(self, **kwargs):
        return

    def is_valid_candidate(self, candidate):
        return True

    def is_valid_evaluation(self, evaluation) -> bool:
        return True
