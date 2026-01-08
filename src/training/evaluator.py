import torch   
from torch.nn import Module
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import recall_score, roc_curve,       \
                            precision_score, f1_score,     \
                            accuracy_score, roc_auc_score, \
                            ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
    
def plot_roc_curve(
    y_true: np.ndarray | list, y_pred: np.ndarray | list
) -> None:
    """
    Plot the ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray | list, y_pred: np.ndarray | list
) -> None:
    """
    Plot the confusion matrix
    """
    plt.rc('font', size=9)  # extra code – make the text smaller
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()
    

class TorchEvaluator:
    """
    Utility class for evaluating a PyTorch model
    """
    def __init__(self, device="cpu"):
        self.device = device

        self.best_models = None
        self.best_metrics = -float('inf')
        self.best_params = None
        self.metric_of_interest = accuracy_score
   
    # 2 class -> one ouput -> sigmoid 
    # 2 class -> 2 outputs -> softmax

    def best_model_params(self):
        """
        Return the best model and its parameters
        """
        return self.best_models, self.best_params

    def evaluate(
        self, model: Module, val_loader: DataLoader, cfg: dict
    ) -> float:
        """
        Evaluate the model on the validation set during searching and update best configs
        """
        model.eval()

        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                probs = torch.sigmoid(model(inputs)).squeeze().cpu()
                outputs_binary = (probs >= 0.5).float()

                # Ensure shapes match by squeezing/reshaping if needed
                labels = labels.squeeze().cpu()

                y_true.append(labels)
                y_pred.append(outputs_binary)

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()

        curr_metrics = self.metric_of_interest(y_true, y_pred)
        if curr_metrics > self.best_metrics:
            self.best_metrics = curr_metrics
            self.best_models = model
            self.best_params = cfg
       
        return curr_metrics

        

class FinalEvaluator:
    """
    Utility class for evaluating a final model on test set
    """
    _registry_predict = {}
    _registry_predict_proba = {}

    def __init__(self, device="cpu"):
        self.device = device

    @staticmethod
    def get_all_metrics(
        y_true: np.ndarray | list, y_pred: np.ndarray | list
    ) -> pd.DataFrame:
        """
        Get all metrics for the model
        """
        return pd.DataFrame({
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_true, y_pred), 4)
        }, index=[0])


    def torch_predict(
        self, model: Module, dataloader: DataLoader, proba: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the model on the test set and return true and predicted labels
        """
        model.eval()
        y_preds = []
        y_trues = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)

                probs = torch.sigmoid(outputs).squeeze(-1).cpu()
                if not proba:
                    probs = (probs >= 0.5).float()
                y_preds.append(probs)
                y_trues.append(targets.cpu())
        
        y_true = torch.cat(y_trues).flatten().numpy()
        y_pred = torch.cat(y_preds).flatten().numpy()
        return y_true, y_pred

    # ---- Register predict ----
    @classmethod
    def register_predict(cls, model_cls):
        def decorator(func):
            cls._registry_predict[model_cls] = func
            return func
        return decorator

    def predict(self, model, x):
        for cls, fn in self._registry_predict.items():
            if isinstance(model, cls):
                return fn(self, model, x)
        raise ValueError(f"No registered predict() for {type(model)}")


    @classmethod
    def register_predict_proba(cls, model_cls):
        def decorator(func):
            cls._registry_predict_proba[model_cls] = func
            return func
        return decorator

    def predict_proba(self, model, x):
        for cls, fn in self._registry_predict_proba.items():
            if isinstance(model, cls):
                return fn(self, model, x)
        raise ValueError(f"No registered predict_proba() for {type(model)}")



# bellow, run at import time !!
# Functions bellow are registered in _registry_predict and _registry_predict_proba

# so, for example: FinalEvaluator._registry_predict[BaseEstimator] = _

# ---- Register sklearn ----
@FinalEvaluator.register_predict(BaseEstimator)
def _(self, model, data):
    X, y_true = data
    return y_true, model.predict(X)

@FinalEvaluator.register_predict_proba(BaseEstimator)
def _(self, model, data):
    X, y_true = data
    return y_true, model.predict_proba(X)


# ---- Register torch ----
@FinalEvaluator.register_predict(torch.nn.Module)
def _(self, model, dataloader):
    return self.torch_predict(model, dataloader)

@FinalEvaluator.register_predict_proba(torch.nn.Module)
def _(self, model, dataloader):
    return self.torch_predict(model, dataloader, proba=True)
