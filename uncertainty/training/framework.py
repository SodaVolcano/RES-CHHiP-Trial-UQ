"""
List of functions for training, saving, and evaluating model
"""

from itertools import islice
from typing import Callable
from arrow import get
from loguru import logger
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    n_batches: int,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,  # type: ignore
):
    """
    Train the model on n_iter batches of an infinite DataLoader object

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing infinite stream of training data
    model : nn.Module
        Model to train
    n_batches : int
        Number of batches to train on for one epoch
    loss_fn : nn.Module
        Loss function
    optimiser : torch.optim.Optimizer
        Optimiser
    """
    device = get_device()

    model.to(device)
    model.train()
    pbar = tqdm(islice(dataloader, n_batches), total=n_batches)
    for data in pbar:
        X, y = data[0].to(device), data[1].to(device)

        optimiser.zero_grad()  # zero gradients for every batch
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimiser.step()

        pbar.set_description(f"loss: {loss.item():.4f}")


def train(
    dataloader: DataLoader,
    model: nn.Module,
    epochs: int,
    n_batches_per_epoch: int,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,  # type: ignore
    tolerance: float = 1e-3,
):
    """
    Train a model for a number of epochs

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing training data
    model : nn.Module
        Model to train
    epochs : int
        Number of epochs to train for
    n_batches_per_epoch : int
        Number of batches to train on for one epoch
    loss_fn : nn.Module
        Loss function
    optimiser : torch.optim.Optimizer
        Optimiser used to update model parameters
    tolerance : float
        Minimum improvement in loss to continue training
    """
    logger.info(f"Training on {get_device()}")

    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(dataloader, model, n_batches_per_epoch, loss_fn, optimiser)
        # validate on new set, 1:5 of training, get avg loss of last N batches for training and valid
        # early stopping
        # save best model and at every N epochs
        test(dataloader, model, loss_fn)
    logger.info("Done!")


def test(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    device = get_device()
    model.eval()

    test_loss, correct = 0, 0
    # Disable gradient computation and reduce memory consumption
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    logger.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def save_model(model: nn.Module, path: str = "./checkpoints/model.pth"):
    torch.save(model.state_dict(), path)


def load_model(
    model_fn: Callable[..., nn.Module], path: str = "./checkpoints/model.pth"
):
    """
    Load a model from a file with evaluation mode
    """
    model = model_fn().to(get_device())
    model.load_state_dict(torch.load(path))
    return model
