from configs.config import RESULTS_DIR
import random
import os
import yaml
import pandas as pd
from pathlib import Path

def sample_config(search_space_dict: dict) -> None | dict:
    """
    Recursively sample from a nested search space dictionary, preserving structure.
    Example: {"lstm": {"input_size": [10], "hidden_size": [128, 256]}}  
        -> {"lstm": {"input_size": 10, "hidden_size": 256}}
    """
    if search_space_dict is None:
        return None
    sampled = {}
    for key, value in search_space_dict.items():
        if isinstance(value, dict):
            # Recursively sample nested dictionaries
            sampled[key] = sample_config(value)
        elif isinstance(value, list):
            # Sample from list of values
            sampled[key] = random.choice(value)
        else:
            # Keep as is (non-list values)
            sampled[key] = value
    return sampled


def save_yaml_config(path_dir: Path | str, **config: dict) -> None:
    """
    Save the configuration model parameters to a YAML file.
    """
    path_dir = Path(path_dir)
    os.makedirs(path_dir, exist_ok=True)
    yaml.dump(config, open(os.path.join(path_dir, "config.yaml"), "w"))


def save_metrics(metrics: pd.DataFrame) -> None:
    """
    Save the metrics to a CSV file.
    """
    path = RESULTS_DIR / "metrics.csv"
    # Append values; write header only when file is new/empty
    write_header = not path.exists() or path.stat().st_size == 0
    metrics.to_csv(path, mode="a", index=False, header=write_header)