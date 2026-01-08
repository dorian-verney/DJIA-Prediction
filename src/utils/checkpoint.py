import os
import pickle
from pathlib import Path
import torch
from torch.nn import Module
from sklearn.base import BaseEstimator

from configs.config import CHECKPOINTS_DIR
from src.utils.utils import save_yaml_config


def _resolve_model_path(path_dir: os.PathLike | str) -> Path:
    """
    Return the checkpoint file path, supporting .pt and .pkl
    """
    base = Path(path_dir)
    pt_path = base / "model.pt"
    pkl_path = base / "model.pkl"

    if pt_path.exists():
        return pt_path
    if pkl_path.exists():
        return pkl_path
    raise FileNotFoundError(f"No checkpoint found in {base} " \
                            f"(expected model.pt or model.pkl)")


def save_checkpoint(model: Module | BaseEstimator, path_dir: Path | str) -> None:
    """
    Save the model checkpoint to the specified path
    """
    path_dir = Path(path_dir)
    path_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "state_dict"):  # PyTorch model
        # TODO: save optimizer state as well
        torch.save(model.state_dict(), path_dir / "model.pt")
    else:  # sklearn model or others
        with open(path_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)


def load_checkpoint(
    path_dir: Path, model_class: Module = None, config: dict | None = None
) -> Module | BaseEstimator:
    """
    Load the model checkpoint from the specified path
    """
    model_path = _resolve_model_path(path_dir)

    if model_path.suffix == ".pt":
        model = model_class(config)
        model.load_state_dict(torch.load(model_path))
        return model
    else:
        with open(model_path, "rb") as f:
            return pickle.load(f)


def save_model(
    best_model: Module | BaseEstimator, best_params: dict, current_datetime: str
) -> None:
    """
    Save the best model checkpoint and configuration
    """
    model_name = best_model.__class__.__name__
    checkpoint_path = CHECKPOINTS_DIR / f"{current_datetime}_{model_name}"
    save_checkpoint(best_model, checkpoint_path)
    save_yaml_config(
        checkpoint_path, **{"model": model_name, "params": best_params}
    )
