from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim import lr_scheduler

from torch.nn import Module

class Trainer:
    """
    Utility class for training, validating and testing a PyTorch model
    """
    def __init__(
        self, model: Module, train_loader: DataLoader, val_loader: DataLoader, 
        criterion: nn, optimizer: Optimizer, device: str, 
        scheduler: lr_scheduler = ReduceLROnPlateau) -> None:
 
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.device       = device
        self.scheduler    = scheduler

        self.model.to(device)

    def calc_loss_batch(
        self, input_batch: Tensor, target_batch: Tensor
    ) -> Tensor:
        """
        Calculate the loss for a batch of data
        """
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = self.criterion(logits, target_batch)
        return loss


    def calc_loss_loader(
        self, data_loader: DataLoader, num_batches: int | None = None
    ) -> float:
        """
        Calculate the loss for a loader of data
        """
        total_loss = 0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(
                    input_batch, target_batch
                )
                total_loss += (loss.item() * input_batch.size(0))
            else:
                break
        return total_loss / num_batches


    def train(
        self, epochs: int, patience: int = 5
    ) -> tuple[list[float], list[float], int]:
        """
        Train the model for a given number of epochs, with early stopping
        """
        train_losses = []
        val_losses = []
        val_loss_max = float('inf')
        patience_counter = 0
        samples_seen = 0
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.calc_loss_batch(
                    input_batch, target_batch
                )
                loss.backward()
                self.optimizer.step()
                samples_seen += input_batch.shape[0]

                train_loss += (loss.item() * input_batch.size(0))
            train_loss = train_loss / len(self.train_loader)
            train_losses.append(train_loss)   
            
            # Validation
            self.model.eval()
            val_loss = self.calc_loss_loader(self.val_loader)
            val_losses.append(val_loss)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
                
            print(f"Epoch {epoch+1}: "
                  f"Train loss {train_loss:.3f}, "
                  f"Val loss {val_loss:.3f}")

            # Early stopping
            if val_loss < val_loss_max:
                val_loss_max = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping due to no improvement "
                      f"after {patience} epochs")
                break

        return train_losses, val_losses, samples_seen
    
    
    def get_trained_model(self):
        """
        Return the trained model
        """
        return self.model