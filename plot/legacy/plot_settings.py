baseline_mapping = {
    "RS": {
        "color": "#c12f2f",  # Dark navy blue
        "label": "RS",
        "marker": "o",
        "linestyle": "-",
    },
    "RSBO": {
        "color": "#d35400",  # Darker burnt orange
        "label": "RSBO",
        "marker": "o",
        "linestyle": "-",
    },
    "LSBO": {
        "color": "#8b0000",  # Dark red
        "label": "LSBO",
        "marker": "s",
        "linestyle": "--",
    },
    "NSGAII": {
        "color": "#006400",  # Dark green
        "label": "NSGAII",
        "marker": "^",
        "linestyle": "-.",
    },
}

mooLLM_mapping = {
    "parallel-no-metrics": {
        "color": "#5a3d7c",  # Darker purple
        "label": "mooLLM (Parallel CS, Direct Optimization)",
        "marker": "o",
        "linestyle": "-",
    },
    "parallel-target-metrics": {
        "color": "#5b3a29",  # Dark earth brown
        "label": "mooLLM (Parallel CS, Desired Target Value)",
        "marker": "s",
        "linestyle": "--",
    },
    "parallel-surrogate-no-metrics": {
        # "color": "#8b0000",  # Darker raspberry pink
        "color": "#003365",
        # "label": "mooLLM (Parallel CS + SM, Direct Optimization)",
        "label": "mooLLM (GPT-4o Mini)",
        "marker": "^",
        "linestyle": "-.",
    },
    "sequential-no-metrics": {
        "color": "#d35400",  # Dark gray
        "label": "mooLLM (Sequential, Direct Optimization)",
        "marker": ">",
        "linestyle": "-",
    },
    "sequential-target-metrics": {
        "color": "#7f8c15",  # Dark olive green
        "label": "mooLLM (Sequential, Desired Target Value)",
        "marker": "d",
        "linestyle": "-",
    },
}

mooLLM_zdt_mapping = {
    "zdt3": {
        "color": "#5a3d7c",  # Darker purple
        "label": "mooLLM (GPT-4o Mini)",
        "marker": "o",
        "linestyle": "-",
    },
}

mooLLM_prompt_comparison_mapping = {
    "loss": {
        "color": "#007a91",  # Dark cyan
        "label": "mooLLM (Loss)",
        "marker": "o",
        "linestyle": "-",
    },
    "error": {
        "color": "#6c8ea4",  # Darker light blue
        "label": "mooLLM (Error)",
        "marker": "s",
        "linestyle": "--",
    },
    "performance": {
        "color": "#5fa163",  # Darker pale green
        "label": "mooLLM (Performance)",
        "marker": "^",
        "linestyle": "-.",
    },
    "generic": {
        "color": "#d9773c",  # Dark peach/orange
        "label": "mooLLM (Generic)",
        "marker": "x",
        "linestyle": ":",
    },
}

mooLLM_model_comparison_mapping = {
    "gemini": {
        "color": "#409425",  # Dark salmon red
        # "color": "#003365",  # Dark salmon red
        # "color": "#2a6144",
        "label": "mooLLM (Gemini)",
        "marker": "o",
        "linestyle": "-",
    },
    "deepseek": {
        "color": "#8e44ad",  # Dark lavender/purple
        "label": "mooLLM (DeepSeek R1)",
        "marker": ">",
        "linestyle": "--",
    },
    "gpt4o-mini": {
        "color": "#8e7c70",  # Dark warm gray
        "label": "mooLLM (GPT-4o Mini)",
        "marker": "^",
        "linestyle": "-.",
    },
    "gpt4o-mini-space-partitioning": {
        "color": "#f57b42",  # Green
        "label": "mooLLM-Voronoi (GPT-4o Mini)",
        "marker": "<",
        "linestyle": "-",
    },
    "Qwen2.5-7B-Instruct-AWQ": {
        "color": "#e024be",  # Dark pale green
        "label": "mooLLM-Voronoi (Qwen2.5-7B)",
        "marker": "*",
        "linestyle": "-",
    },
    "QwQ-32B-AWQ": {
        "color": "#cc1f9e",
        "label": "mooLLM (QwQ-32B-AWQ)",
        "marker": "s",
        "linestyle": ":",
    },
    "QwQ-32B-AWQ-ToT": {
        "color": "#cc1f9e",
        "label": "mooLLM-ToT (QwQ-32B-AWQ)",
        "marker": ">",
        "linestyle": "-",
    },
}

mooLLM_tot_mapping = {
    "tot": {
        "color": "#b8860b",  # Firebrick red
        # "label": "mooLLM (ToT)",
        "label": "mooLLM-ToT (GPT-4o Mini)",
        "marker": ">",
        "linestyle": "-",
    },
    "tot-checkpoint": {
        "color": "#b8860b",  # Dark goldenrod
        "label": "mooLLM (ToT Checkpoint)",
        "marker": "s",
        "linestyle": "--",
    },
    "gemini-tot": {
        "color": "#eee420",  # Dark warm gray
        "label": "mooLLM-ToT (Gemini)",
        "marker": "^",
        "linestyle": "-.",
    },
    "gemini-tot-alternate": {
        # "color": "#d03cd7",
        "color": "#469e20",
        "label": "mooLLM-ToT (Gemini, I/O and ToT alternating)",
        "marker": ">",
        "linestyle": "-.",
    },
}

moLLM_deepseek_mapping = {
    "deepseek": {
        "color": "#8e44ad",
        "label": "mooLLM (I/O)",
        "marker": "s",
        "linestyle": "--",
    },
    "deepseek_tot": {
        "color": "#b8860b",
        "label": "mooLLM (ToT)",
        "marker": ">",
        "linestyle": "--",
    },
    "deepseek_tot_mooLLM_warmstarting": {
        "color": "#8e7c70",
        "label": "mooLLM (I/O first + ToT)",
        "marker": "*",
        "linestyle": "--",
    },
    "deepseek_tot_mooLLM_alternate": {
        # "color": "#d9773c",
        "color": "#ad4490",
        "label": "mooLLM-ToT (DeepSeek R1, I/O and ToT alternating)",
        "marker": ">",
        "linestyle": "--",
    },
    "deepseek-orig": {
        "color": "#5fa163",
        "label": "mooLLM (DeepSeek R1 (Original))",
        "marker": "^",
        "linestyle": "--",
    },
}


def get_visuals(baseline_key):
    """
    Get the visual settings for the model.
    """
    print("THIS BASELINE KEY: ", baseline_key)
    if "RSBO" in baseline_key:
        return baseline_mapping["RSBO"]
    elif "RS" in baseline_key:
        return baseline_mapping["RS"]
    elif "LSBO" in baseline_key:
        return baseline_mapping["LSBO"]
    elif "NSGA2" in baseline_key:
        return baseline_mapping["NSGAII"]
    elif "gpt-4o-mini_zdt_zdt3" in baseline_key:
        return mooLLM_zdt_mapping["zdt3"]
    elif "parallel-no-metrics" in baseline_key:
        return mooLLM_mapping["parallel-no-metrics"]
    elif "parallel-target-metrics" in baseline_key:
        return mooLLM_mapping["parallel-target-metrics"]
    elif (
        "parallel-surrogate-no-metrics" in baseline_key
        or "gpt-4o-mini_latencies" in baseline_key
    ):
        print("here")
        return mooLLM_mapping["parallel-surrogate-no-metrics"]
    elif "sequential-no-metrics" in baseline_key:
        return mooLLM_mapping["sequential-no-metrics"]
    elif "sequential-target-metrics" in baseline_key:
        return mooLLM_mapping["sequential-target-metrics"]
    elif "loss" in baseline_key:
        return mooLLM_prompt_comparison_mapping["loss"]
    elif "error" in baseline_key:
        return mooLLM_prompt_comparison_mapping["error"]
    elif "performance" in baseline_key:
        return mooLLM_prompt_comparison_mapping["performance"]
    elif "generic" in baseline_key:
        return mooLLM_prompt_comparison_mapping["generic"]
    elif "gemini-1-5-flash-no-context-tot-alternate" in baseline_key:
        return mooLLM_tot_mapping["gemini-tot-alternate"]
    elif "gemini-1-5-flash-tot-interval" in baseline_key:
        return mooLLM_tot_mapping["gemini-tot-alternate"]
    elif "gemini-1-5-flash-tot" in baseline_key:
        return mooLLM_tot_mapping["gemini-tot"]
    elif "deepseek-tot-alternate" in baseline_key:
        print("please")
        return moLLM_deepseek_mapping["deepseek_tot_mooLLM_alternate"]
    elif "tot" in baseline_key:
        return mooLLM_tot_mapping["tot"]
    elif "tot-checkpoint" in baseline_key:
        return mooLLM_tot_mapping["tot-checkpoint"]
    elif "gemini" in baseline_key:
        print("here geminig")
        return mooLLM_model_comparison_mapping["gemini"]
    # elif "deepseek-tot-mooLLM-warmstarting" in baseline_key:
    #     return moLLM_deepseek_mapping["deepseek_tot_mooLLM_warmstarting"]
    # elif "deepseek-tot" in baseline_key:
    #     return moLLM_deepseek_mapping["deepseek_tot"]
    # elif "deepseek-orig" in baseline_key:
    #     return moLLM_deepseek_mapping["deepseek-orig"]
    # elif "deepseek" in baseline_key:
    #     return moLLM_deepseek_mapping["deepseek"]
    elif "QwQ-32B-AWQ-ToT" in baseline_key:
        return mooLLM_model_comparison_mapping["QwQ-32B-AWQ-ToT"]
    elif "QwQ-32B-AWQ" in baseline_key:
        return mooLLM_model_comparison_mapping["QwQ-32B-AWQ"]
    elif "deepseek" in baseline_key:
        print("here deepseek")
        return mooLLM_model_comparison_mapping["deepseek"]
    elif "gpt-4o-mini-space-partitioning" in baseline_key:
        return mooLLM_model_comparison_mapping["gpt4o-mini-space-partitioning"]
    elif "Qwen2.5-7B-Instruct-AWQ" in baseline_key:
        return mooLLM_model_comparison_mapping["Qwen2.5-7B-Instruct-AWQ"]
    elif "gpt4o-mini" in baseline_key:
        print("here gpt4o-mini")
        return mooLLM_model_comparison_mapping["gpt4o-mini"]
    elif "QwQ" in baseline_key:
        return mooLLM_model_comparison_mapping["QwQ"]
