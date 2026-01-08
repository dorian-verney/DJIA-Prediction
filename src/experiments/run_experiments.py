import yaml
from pathlib import Path
from datetime import datetime

from .search import search
from src.data.data_loader import load_dataset
from src.training.evaluator import FinalEvaluator
from src.data.data_loader import to_dataloader
from src.utils.checkpoint import save_model
from src.utils.utils import save_metrics


def run_experiments(config_path: Path | str) -> None:
    """
    Run the experiments for the given configuration path.
    """
    config_path = Path(config_path) 
    # Check if config_path is a file or folder
    if config_path.is_file():
        # If it's a file, run once
        run(config_path)

    elif config_path.is_dir():
        # If it's a folder, iterate over each YAML file and execute run
        for config_file in config_path.glob("*.yaml"):
            run(config_file)
    else:
        raise ValueError(f"Config path '{config_path}' "
                         f"is neither a file nor a directory.")
    
    
def run(config_path: Path) -> None:
    """
    Search the best model among configs, evaluate and save it.
    """
    config_path = Path(config_path)  # Convert string to Path object
    cfg = yaml.safe_load(open(config_path))
    ds_train, ds_test = load_dataset(cfg["dataset"], 
                                     test_size=0.2,
                                     transform=True)

    # 1. search for best model
    print(f"\n\nRunning model: {cfg['model']}\n")
    if "search_args" in cfg and cfg["search_args"]["type"] == "random":
        best_model, best_params = search(
            cfg,
            ds_train,
        )
        print("Best Model:", best_model)
        print("Best Params:", best_params)
    
    # 2. evaluate best model
    evaluator = FinalEvaluator()
    if cfg["library"] == "torch":
        X_test, y_test = ds_test
        ds_test = to_dataloader(
            X_test, y_test, cfg["search_space"]["training"], batch_size=16
        )
    y_true, y_pred = evaluator.predict(best_model, ds_test)
    metrics = evaluator.get_all_metrics(y_true, y_pred)

    # 3. save best model as checkpoint
    current_datetime = datetime.now().strftime('%m-%d_%H-%M-%S')
    save_model(best_model, best_params, current_datetime)

    # 4. save metrics
    metrics["model"] = cfg["model"]
    metrics["datetime"] = current_datetime
    save_metrics(metrics)



if __name__ == "__main__":
    import sys
    run_experiments(sys.argv[1], sys.argv[2])
