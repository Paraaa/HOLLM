import logging
from typing import Dict
from mooLLM.llm.llm import LLMInterface
from mooLLM.llm.models.local import LOCAL
from mooLLM.llm.models.gpt import GPT
from mooLLM.llm.models.gemini import GEMINI
from mooLLM.llm.models.groq import GROQ
from mooLLM.llm.models.huggingface import HUGGINGFACE
from mooLLM.settings import (
    MODELS,
    ACQUISITION_FUNCTIONS,
    SURROGATE_MODELS,
    CANDIDATE_SAMPLERS,
    WARMSTARTERS,
    CONTEXT_LIMIT_STRATEGIES,
    SCHEDULERS,
    SPACE_PARTITIONING_STRATEGIES,
    REGION_ACQUISITION_FUNCTIONS,
)
from mooLLM.optimization_strategy.optimization_strategy import OptimizationStrategy
from mooLLM.optimization_strategy.mooLLM import mooLLM
from mooLLM.optimization_strategy.space_partitioning_mooLLM import (
    SpacePartitioningMOOLLM,
)
from mooLLM.benchmarks.benchmark import BENCHMARK
from mooLLM.acquisition_functions.acquisition_function import ACQUISITION_FUNCTION
from mooLLM.region_acquisition_functions.region_acq import RegionACQ
from mooLLM.surrogate_models.surrogate_model import (
    SURROGATE_MODEL,
)
from mooLLM.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER
from mooLLM.warmstarter.warmstarter import WARMSTARTER
from mooLLM.statistics.context_limit_strategy.context_limit_strategy import (
    ContextLimitStrategy,
)

from mooLLM.space_partitioning.space_partitioning_strategy import (
    SPACE_PARTITIONING_STRATEGY,
)
from mooLLM.space_partitioning.utils import BoundingBox

from mooLLM.statistics.statistics import Statistics
from mooLLM.utils.rate_limiter import RateLimiter
from mooLLM.utils.prompt_builder import PromptBuilder

logger = logging.getLogger("Builder")


class Builder:
    def __init__(
        self,
        config: Dict,
        benchmark: BENCHMARK,
        custom_model: LLMInterface = None,
        custom_acquisition_function: ACQUISITION_FUNCTION = None,
        custom_candidate_sampler: CANDIDATE_SAMPLER = None,
        custom_space_partitioning_strategy: SPACE_PARTITIONING_STRATEGY = None,
        custom_warmstarter: WARMSTARTER = None,
        custom_surrogate_model: SURROGATE_MODEL = None,
    ) -> None:
        logger.debug(f"Setting up builder with configuration: {config}")

        # Store custom implementations
        self._custom_model = custom_model
        self._custom_acquisition_function = custom_acquisition_function
        self._custom_candidate_sampler = custom_candidate_sampler
        self._custom_space_partitioning_strategy = custom_space_partitioning_strategy
        self._custom_warmstarter = custom_warmstarter
        self._custom_surrogate_model = custom_surrogate_model

        # Model related attributes
        self.llm_settings: dict = config.get("llm_settings", {})

        # Configuration parameters
        self.optimization_method = config.get("optimization_method", "mooLLM")
        self.tot_settings = config.get("tot_settings", {})
        self.interval_settings = config.get("interval_settings", {})
        self.alpha = config.get("alpha", -0.2)
        self.n_trials = config.get("n_trials", 5)
        self.total_trials = config.get("total_trials", 5)
        self.max_candidates_per_trial = config.get("max_candidates_per_trial", 5)
        self.candidates_per_request = config.get("candidates_per_request", 5)
        self.evaluations_per_request = config.get("evaluations_per_request", 5)
        self.max_evaluations_per_trial = config.get("max_evaluations_per_trial", 5)
        self.space_partitioning_settings = config.get("space_partitioning_settings", {})
        self.initial_samples = config.get("initial_samples", 5)
        self.seed = config.get("seed", 42)
        self.max_context_configs = config.get("max_context_configs", 100)
        self.max_requests_per_minute = config.get("max_requests_per_minute", 700)
        self.context_limit_strategy = self.get_context_limit_strategy(
            config.get("context_limit_strategy", None)
        )
        self.shuffle_icl_columns = config.get("shuffle_icl_columns", False)
        self.shuffle_icl_rows = config.get("shuffle_icl_rows", False)
        self.metrics = config.get("metrics", [])
        self.metrics_targets = config.get("metrics_targets", [])
        self.range_parameter_keys = config.get("range_parameter_keys", [])
        self.integer_parameter_keys = config.get("integer_parameter_keys", [])
        self.float_parameter_keys = config.get("float_parameter_keys", [])
        self.parameter_constraints: dict = config.get("parameter_constraints", {})

        self.benchmark_name: str = config.get("benchmark", "")
        self.method_name: str = config.get("method_name", " ")
        self.use_few_shot_examples = config.get("use_few_shot_examples", False)

        self.prompt = config.get("prompt", {})

        self.model_name: str = self.llm_settings.get("model", None)
        self.benchmark: BENCHMARK = benchmark
        self.benchmark.method_name = self.method_name

        self.statistics: Statistics = self.get_statistics()
        self.model: LLMInterface = self.get_model(self.model_name)
        self.warmstarting_prompt_template_dir: str = config.get(
            "warmstarting_prompt_template", None
        )
        self.candidate_sampler_prompt_template_dir: str = config.get(
            "candidate_sampler_prompt_template", None
        )
        self.surrogate_model_prompt_template_dir: str = config.get(
            "surrogate_model_prompt_template", None
        )

        self.acquisition_function: ACQUISITION_FUNCTION = self.get_acquisition_function(
            config.get("acquisition_function", "")
        )
        self.surrogate_model: SURROGATE_MODEL = self.get_surrogate_model(
            config.get("surrogate_model", "")
        )
        self.candidate_sampler: CANDIDATE_SAMPLER = self.get_candidate_sampler(
            config.get("candidate_sampler", "")
        )
        self.warmstarter: WARMSTARTER = self.get_warmstarter(
            config.get("warmstarter", "")
        )

        logger.debug("Setting up builder complete")

    def get_candidate_sampler(self, candidate_sampler: str) -> CANDIDATE_SAMPLER:
        """
        Retrieve and configure a candidate sampler based on the provided name.

        This method first checks if a custom candidate sampler was provided at initialization.
        If not, it looks up the candidate sampler specified by the 'candidate_sampler'
        argument in the CANDIDATE_SAMPLERS dictionary. If found, it configures the sampler
        with the necessary attributes from the current object's context.

        Args:
            candidate_sampler (str): The name of the candidate sampler to retrieve.

        Returns:
            CANDIDATE_SAMPLER: An instance of the configured candidate sampler.

        """
        # Check if a custom candidate sampler was provided
        if self._custom_candidate_sampler is not None:
            logger.debug("Using custom candidate sampler")
            selected_candidate_sampler = self._custom_candidate_sampler
        else:
            selected_candidate_sampler = CANDIDATE_SAMPLERS.get(candidate_sampler, None)
            if selected_candidate_sampler is None:
                logger.warning(f"No candidate sampler specified: {candidate_sampler}")
                return

        selected_candidate_sampler.max_candidates_per_trial = (
            self.max_candidates_per_trial
        )
        selected_candidate_sampler.candidates_per_request = self.candidates_per_request
        selected_candidate_sampler.model = self.model
        selected_candidate_sampler.statistics = self.statistics
        selected_candidate_sampler.alpha = self.alpha
        selected_candidate_sampler.metrics = self.metrics
        selected_candidate_sampler.benchmark = self.benchmark
        selected_candidate_sampler.tot_settings = self.tot_settings
        selected_candidate_sampler.prompt_builder = self.get_prompt_builder(
            template_dir=self.candidate_sampler_prompt_template_dir,
        )
        selected_candidate_sampler.range_parameter_keys = self.range_parameter_keys
        return selected_candidate_sampler

    def get_warmstarter(self, warmstarter: str) -> WARMSTARTER:
        """
        Retrieves and configures a warmstarter based on the provided name.

        This method first checks if a custom warmstarter was provided at initialization.
        If not, it looks up the warmstarter specified by the 'warmstarter'
        argument in the WARMSTARTERS dictionary. If found, it configures the
        warmstarter with the necessary attributes from the current object's context.

        Args:
            warmstarter (str): The name of the warmstarter to retrieve.

        Returns:
            WARMSTARTER: An instance of the configured warmstarter.

        """
        # Check if a custom warmstarter was provided
        if self._custom_warmstarter is not None:
            logger.debug("Using custom warmstarter")
            selected_warmstarter = self._custom_warmstarter
        else:
            selected_warmstarter = WARMSTARTERS.get(warmstarter, None)
            if selected_warmstarter is None:
                logger.warning(f"No warmstarter specified: {warmstarter}")
                return

        selected_warmstarter.model = self.model
        selected_warmstarter.initial_samples = self.initial_samples
        selected_warmstarter.benchmark = self.benchmark
        selected_warmstarter.prompt_builder = self.get_prompt_builder(
            template_dir=self.warmstarting_prompt_template_dir,
        )

        return selected_warmstarter

    def get_acquisition_function(
        self, acquisition_function: str
    ) -> ACQUISITION_FUNCTION:
        """
        Retrieves and configures an acquisition function based on the provided name.

        This method first checks if a custom acquisition function was provided at initialization.
        If not, it looks up the acquisition function specified by the
        'acquisition_function' argument in the ACQUISITION_FUNCTIONS dictionary. If
        found, it configures the acquisition function with the necessary attributes
        from the current object's context.

        Args:
            acquisition_function (str): The name of the acquisition function to
                retrieve.

        Returns:
            ACQUISITION_FUNCTION: An instance of the configured acquisition
                function.

        """
        # Check if a custom acquisition function was provided
        if self._custom_acquisition_function is not None:
            logger.debug("Using custom acquisition function")
            selected_acquisition_function = self._custom_acquisition_function
        else:
            selected_acquisition_function: ACQUISITION_FUNCTION = (
                ACQUISITION_FUNCTIONS.get(acquisition_function, None)
            )
            if selected_acquisition_function is None:
                logger.warning(
                    f"No acquisition function specified: {acquisition_function}"
                )
                return

        selected_acquisition_function.statistics = self.statistics
        selected_acquisition_function.metrics_targets = self.metrics_targets
        return selected_acquisition_function

    def get_surrogate_model(self, surrogate_model: str) -> SURROGATE_MODEL:
        """
        Retrieves and configures a surrogate model based on the provided name.

        This method first checks if a custom surrogate model was provided at initialization.
        If not, it looks up the surrogate model specified by the
        'surrogate_model' argument in the SURROGATE_MODELS dictionary. If
        found, it configures the surrogate model with the necessary attributes
        from the current object's context.

        Args:
            surrogate_model (str): The name of the surrogate model to
                retrieve.

        Returns:
            SURROGATE_MODEL: An instance of the configured surrogate
                model.

        """
        # Check if a custom surrogate model was provided
        if self._custom_surrogate_model is not None:
            logger.debug("Using custom surrogate model")
            selected_surrogate_model = self._custom_surrogate_model
        else:
            selected_surrogate_model: SURROGATE_MODEL = SURROGATE_MODELS.get(
                surrogate_model, None
            )
            if selected_surrogate_model is None:
                logger.warning(f"No surrogate model specified: {surrogate_model}")
                return

        selected_surrogate_model.model = self.model
        selected_surrogate_model.statistics = self.statistics
        selected_surrogate_model.benchmark = self.benchmark
        selected_surrogate_model.max_evaluations_per_trial = (
            self.max_evaluations_per_trial
        )
        selected_surrogate_model.evaluations_per_request = self.evaluations_per_request
        logger.debug(f"Template dir: {self.surrogate_model_prompt_template_dir}")

        selected_surrogate_model.prompt_builder = self.get_prompt_builder(
            template_dir=self.surrogate_model_prompt_template_dir,
        )

        return selected_surrogate_model

    def get_prompt_builder(self, template_dir: str) -> PromptBuilder:
        """
        Retrieves and configures a PromptBuilder based on the provided template directory.

        Args:
            template_dir (str): The directory containing the prompt template.

        Returns:
            PromptBuilder: An instance of the configured PromptBuilder.
        """
        prompt_builder: PromptBuilder = PromptBuilder(template_dir=template_dir)
        prompt_builder.initial_samples = self.initial_samples
        prompt_builder.statistics = self.statistics
        prompt_builder.shuffle_icl_columns = self.shuffle_icl_columns
        prompt_builder.shuffle_icl_rows = self.shuffle_icl_rows
        prompt_builder.use_few_shot_examples = self.use_few_shot_examples
        prompt_builder.few_shot_examples = self.benchmark.get_few_shot_samples()
        prompt_builder.metrics_ranges = self.benchmark.get_metrics_ranges()
        prompt_builder.prompt = self.prompt
        prompt_builder.parameter_constraints = self.parameter_constraints

        prompt_builder.feature_names = self.parameter_constraints.keys()
        prompt_builder.metrics_names = self.metrics
        prompt_builder.metrics_targets = self.metrics_targets
        prompt_builder.initialize_templates()
        return prompt_builder

    def get_context_limit_strategy(
        self, context_limit_strategy: str
    ) -> ContextLimitStrategy:
        selected_context_limit_strategy: ContextLimitStrategy = (
            CONTEXT_LIMIT_STRATEGIES.get(context_limit_strategy, None)
        )
        if selected_context_limit_strategy is None:
            logger.warning(
                f"No context limit strategy specified: {context_limit_strategy}"
            )
            return

        selected_context_limit_strategy.max_context_configs = self.max_context_configs
        return selected_context_limit_strategy

    def get_statistics(self) -> Statistics:
        """
        Retrieves and configures a Statistics object.

        Returns:
            Statistics: An instance of the configured Statistics object.
        """
        statistics = Statistics()
        statistics.model_name = self.model_name
        statistics.seed = self.seed
        statistics.initial_samples = self.initial_samples
        statistics.max_context_configs = self.max_context_configs
        statistics.context_limit_strategy = self.context_limit_strategy
        statistics.metrics = self.metrics
        statistics.benchmark_name = self.benchmark_name
        return statistics

    def get_model(self, model: str) -> LLMInterface:
        """
        Retrieves and configures a model based on the provided model name.
        If a custom model was provided at initialization, it will be used instead.

        Args:
            model (str): The name of the model to retrieve.

        Returns:
            LLMInterface: An instance of the configured model.

        """
        # Check if a custom model was provided
        if self._custom_model is not None:
            logger.debug("Using custom model")
            selected_model = self._custom_model
        else:
            selected_model = MODELS.get(model, None)
            if selected_model is None:
                logger.warning(f"No model specified: {model}")
                return

        # Load the model to memory if it's a LOCAL model.
        # This prevents the model from being reloaded after each prompt.
        if isinstance(selected_model, LOCAL):
            selected_model._load_model_to_memory()
        # If the model is of type GPT, GEMINI, GROQ or HUGGINGFACE, set the rate limiter
        if isinstance(selected_model, (GPT, GEMINI, GROQ, HUGGINGFACE)):
            selected_model.rate_limiter = RateLimiter(
                max_tokens=self.llm_settings.get("max_tokens_per_minute", 10000),
                time_frame=60,
                max_requests=self.llm_settings.get("max_requests_per_minute", 700),
            )
        selected_model.initial_samples = self.initial_samples
        selected_model.statistics = self.statistics
        selected_model.llm_settings = self.llm_settings
        return selected_model

    def get_optimization_method(self) -> OptimizationStrategy:
        optimization_method = None
        logger.debug(f"Optimization method: {self.optimization_method}")
        if self.optimization_method == "mooLLM":
            optimization_method: mooLLM = mooLLM()
            optimization_method.warmstarter = self.warmstarter
            optimization_method.candidate_sampler = self.candidate_sampler
            optimization_method.acquisition_function = self.acquisition_function
            optimization_method.surrogate_model = self.surrogate_model
            optimization_method.statistics = self.statistics
            optimization_method.benchmark = self.benchmark
            optimization_method.initial_samples = self.initial_samples
            optimization_method.n_trials = self.n_trials
            optimization_method.initialize()
            logger.debug("Initialized new mooLLM instance with configured components")
        elif self.optimization_method == "TOT":
            optimization_method: mooLLMToT = mooLLMToT()
            optimization_method.tot_settings = self.tot_settings
            optimization_method.warmstarter = self.warmstarter
            optimization_method.candidate_sampler = self.candidate_sampler
            optimization_method.acquisition_function = self.acquisition_function
            optimization_method.surrogate_model = self.surrogate_model
            optimization_method.statistics = self.statistics
            optimization_method.benchmark = self.benchmark
            optimization_method.initial_samples = self.initial_samples
            optimization_method.n_trials = self.n_trials
            optimization_method.initialize()
            logger.debug(
                "Initialized new mooLLMToT instance with configured components"
            )
        elif self.optimization_method == "Interval":
            optimization_method: mooLLMInterval = mooLLMInterval()
            # Building the mooLLM instance
            mooLLM_instance: mooLLM = mooLLM()
            mooLLM_instance.warmstarter = self.warmstarter
            mooLLM_instance.candidate_sampler = self.candidate_sampler
            mooLLM_instance.acquisition_function = self.acquisition_function
            mooLLM_instance.surrogate_model = self.surrogate_model
            mooLLM_instance.statistics = self.statistics
            mooLLM_instance.benchmark = self.benchmark
            mooLLM_instance.initial_samples = self.initial_samples
            mooLLM_instance.n_trials = self.interval_settings.get("mooLLM_trials", 5)
            mooLLM_instance.initialize()
            logger.debug("Initialized new mooLLM instance with configured components")

            # Building the TOT instance
            mooLLM_tot_instance: mooLLMToT = mooLLMToT()
            mooLLM_tot_instance.tot_settings = self.tot_settings
            mooLLM_tot_instance.warmstarter = None
            mooLLM_tot_instance.n_trials = self.interval_settings.get(
                "mooLLM_tot_trials", 5
            )
            mooLLM_tot_instance.candidate_sampler = self.get_candidate_sampler(
                "TOT_SAMPLER"
            )
            mooLLM_tot_instance.candidate_sampler.prompt_builder = (
                self.get_prompt_builder(
                    template_dir=self.interval_settings.get(
                        "tot_candidate_sampler_template", ""
                    ),
                )
            )
            mooLLM_tot_instance.surrogate_model = self.get_surrogate_model(
                "mooLLM_SUR_BATCH"
            )
            mooLLM_tot_instance.surrogate_model.prompt_builder = (
                self.get_prompt_builder(
                    template_dir=self.interval_settings.get(
                        "tot_surrogate_model_template", ""
                    ),
                )
            )
            mooLLM_tot_instance.acquisition_function = self.get_acquisition_function(
                "HypervolumeImprovementBatch"
            )
            mooLLM_tot_instance.statistics = self.statistics
            mooLLM_tot_instance.benchmark = self.benchmark
            mooLLM_tot_instance.initial_samples = None
            optimization_method.mooLLM = mooLLM_instance
            optimization_method.mooLLM_tot = mooLLM_tot_instance
            optimization_method.interval_settings = self.interval_settings
            optimization_method.n_trials = self.n_trials
            optimization_method.statistics = self.statistics
            return optimization_method
        elif self.optimization_method == "SpacePartitioning":
            optimization_method: SpacePartitioningMOOLLM = SpacePartitioningMOOLLM()
            optimization_method.benchmark = self.benchmark
            optimization_method.statistics = self.statistics
            optimization_method.warmstarter = self.warmstarter
            optimization_method.n_trials = self.n_trials
            optimization_method.acquisition_function = self.acquisition_function
            optimization_method.surrogate_model = self.surrogate_model
            optimization_method.candidate_sampler = self.candidate_sampler
            optimization_method.top_k = self.space_partitioning_settings.get("top_k", 5)
            optimization_method.partitions_per_trial = (
                self.space_partitioning_settings.get("partitions_per_trial", 5)
            )
            optimization_method.use_clustering = self.space_partitioning_settings.get(
                "use_clustering", False
            )

            # Use the correct region acquisition function
            region_acq_strategy = self.space_partitioning_settings.get(
                "region_acquisition_strategy", "VolumeRegionACQ"
            )
            region_acquisition_function = self.get_region_acquisition_function(
                region_acq_strategy
            )
            optimization_method.region_acquisition_function = (
                region_acquisition_function
            )

            # Get the space partitioning strategy using our method
            search_space_partitioning_strategy = self.get_space_partitioning_strategy(
                self.space_partitioning_settings.get("partitioning_strategy", "voronoi")
            )
            search_space_partitioning_strategy.space_partitioning_settings = (
                self.space_partitioning_settings
            )

            optimization_method.search_space_partitioning = (
                search_space_partitioning_strategy
            )
            optimization_method.use_pareto_front_as_regions = (
                self.space_partitioning_settings.get(
                    "use_pareto_front_as_regions", False
                )
            )

            optimization_method.initialize()
            return optimization_method
        else:
            logger.warning(
                f"No valid optimization method specified: {self.optimization_method}"
            )
            return

        logger.debug(f"Optimization method: {self.optimization_method}")

        return optimization_method

    def build(self):
        """
        Constructs and initializes a mooLLM object with the builder's configurations.

        This method sets up a mooLLM instance by assigning it various components
        such as the warmstarter, candidate sampler, acquisition function, surrogate
        model, statistics, benchmark, initial samples, and number of trials. If
        optimization is to continue from a checkpoint, it adjusts the number of
        trials accordingly. It also calls the initialization method on the mooLLM
        instance to prepare it for optimization.

        Returns:
            mooLLM: The fully constructed and initialized mooLLM object.
        """
        mooLLM: OptimizationStrategy = self.get_optimization_method()
        return mooLLM

    def get_space_partitioning_strategy(
        self, partitioning_strategy: str
    ) -> SPACE_PARTITIONING_STRATEGY:
        """
        Retrieves and configures a space partitioning strategy based on the provided name.

        This method first checks if a custom space partitioning strategy was provided at initialization.
        If not, it looks up the partitioning strategy specified by the 'partitioning_strategy'
        argument in the SPACE_PARTITIONING_STRATEGIES dictionary. If found, it configures the
        strategy with the necessary attributes from the current object's context.

        Args:
            partitioning_strategy (str): The name of the partitioning strategy to retrieve.

        Returns:
            SPACE_PARTITIONING_STRATEGY: An instance of the configured partitioning strategy.
        """
        # Check if a custom space partitioning strategy was provided
        if self._custom_space_partitioning_strategy is not None:
            logger.debug("Using custom space partitioning strategy")
            selected_strategy = self._custom_space_partitioning_strategy
        else:
            selected_strategy = SPACE_PARTITIONING_STRATEGIES.get(
                partitioning_strategy, None
            )
            if selected_strategy is None:
                logger.warning(
                    f"Invalid partitioning strategy specified: {partitioning_strategy}, using Voronoi partitioning"
                )
                selected_strategy = SPACE_PARTITIONING_STRATEGIES.get("voronoi")

        # Configure the bounding box
        bounding_box = BoundingBox(
            volume=0.0,
            boundaries=self.parameter_constraints,
            range_parameter_keys=self.range_parameter_keys,
        )
        bounding_box.calculate_volume()

        # Configure the strategy
        selected_strategy.bounding_box = bounding_box
        selected_strategy.range_parameter_keys = self.range_parameter_keys
        selected_strategy.integer_parameter_keys = self.integer_parameter_keys
        selected_strategy.float_parameter_keys = self.float_parameter_keys
        selected_strategy.statistics = self.statistics

        return selected_strategy

    def get_region_acquisition_function(self, region_acq_function: str) -> RegionACQ:
        """
        Retrieves and configures a region acquisition function based on the provided name.
        This method looks up the region acquisition function in REGION_ACQUISITION_FUNCTIONS.
        """

        selected_region_acq = REGION_ACQUISITION_FUNCTIONS.get(
            region_acq_function, None
        )
        if selected_region_acq is None:
            logger.warning(
                f"No region acquisition function specified: {region_acq_function}"
            )
            return
        # Set any additional attributes here if needed
        selected_region_acq.metrics_targets = self.metrics_targets
        selected_region_acq.space_partitioning_settings = (
            self.space_partitioning_settings
        )
        selected_region_acq.n_trials = self.total_trials
        selected_region_acq.alpha = self.space_partitioning_settings.get(
            "region_acquisition_alpha", 0.5
        )
        # Scheduler setup if needed
        scheduler_name = self.space_partitioning_settings.get(
            "scheduler_settings", {}
        ).get("scheduler")
        if scheduler_name:
            selected_region_acq.scheduler = SCHEDULERS.get(scheduler_name)
            selected_region_acq.scheduler.apply_settings(
                self.space_partitioning_settings.get("scheduler_settings", {})
            )
        return selected_region_acq
