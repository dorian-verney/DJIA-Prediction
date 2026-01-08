from torch.nn import BCEWithLogitsLoss, Module
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.base import BaseEstimator
from src.training.evaluator import TorchEvaluator
from src.utils.utils import sample_config
from src.models.model_registry import MODEL_REGISTRY
from src.training.trainer import Trainer
from src.data.data_loader import to_dataloader

def search(
    config: dict, datasets: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Choose and call the search method according to the model library
    """
    lib = config["library"] 
    match lib:
        case "sklearn":
            best_model = search_sklearn(config, datasets)
        case "torch":
            best_model = search_torch(config, datasets)
        case _:
            raise ValueError(f"Library {lib} not supported")
    
    return best_model


def search_torch(
    config: dict, datasets: tuple[np.ndarray, np.ndarray]
) -> tuple[Module, dict]:
    """
    Search the best model among torch configs, evaluate and save it.
    """
    n_trials = config["search_args"]["n_trials"]
    search_space = config["search_space"]

    X_train, y_train = datasets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, 
                                                          y_train, 
                                                          test_size=0.1, 
                                                          shuffle=False)

    evaluator = TorchEvaluator(device="cpu")

    for _ in range(n_trials):

        # sample a configuration - recursively sample while preserving structure
        model_cfg = sample_config(search_space["model"])
        training_cfg = sample_config(search_space["training"])

        # Get model class and instantiate it
        model = MODEL_REGISTRY[config["model"]](**(model_cfg or {}))
        
        # Get optimizer class and instantiate it
        optimizer = getattr(optim, training_cfg["optimizer"])(
            model.parameters(), **training_cfg["optimizer_args"] or {}
        )
        # Get scheduler class and instantiate it
        scheduler = getattr(optim.lr_scheduler, training_cfg["lr_scheduler"])(
            optimizer, **training_cfg["lr_scheduler_args"] or {}
        )

        train_loader = to_dataloader(X_train, y_train, training_cfg)
        valid_loader = to_dataloader(X_valid, y_valid, training_cfg)

        trainer = Trainer(model=model, 
                          train_loader=train_loader, 
                          val_loader=valid_loader, 
                          criterion=BCEWithLogitsLoss(),
                          optimizer=optimizer, 
                          scheduler=scheduler,
                          device="cpu")

        trainer.train(epochs=config["epochs"])

        # evaluate
        metrics = evaluator.evaluate(
            trainer.get_trained_model(), 
            valid_loader,
            cfg={"model": model_cfg, "training": training_cfg}
        )

        print("-------------------------------- \n")
    
    return evaluator.best_model_params()


def search_sklearn(
    config: dict, datasets: tuple[np.ndarray, np.ndarray]
) -> tuple[BaseEstimator, dict]:
    """
    Search the best model among sklearn configs, evaluate and save it.
    """
    n_trials = config["search_args"]["n_trials"]
    model = MODEL_REGISTRY[config["model"]](config["search_space"]) 
    model_cfg = config["search_space"]

    # distributions = dict(C=uniform(loc=0, scale=4),
    #                     penalty=['l2', 'l1'])

    X_train, y_train = datasets 

    clf = RandomizedSearchCV(
        model,
        model_cfg,
        cv=3,
        n_iter=n_trials, 
        refit=True
    )
    search = clf.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_



