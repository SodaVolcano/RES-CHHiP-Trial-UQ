"""
List of functions for training, saving, and evaluating model
"""

from typing import Callable
from loguru import logger
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def epoch(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimiser: torch.optim.Optimizer):  # type: ignore
    """
    Run a single epoch of training

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing training data
    model : nn.Module
        Model to train
    loss_fn : nn.Module
        Loss function
    optimiser : torch.optim.Optimizer
        Optimiser
    """
    device = get_device()
    logger.info(f"Training on {device}")

    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # prediction loss
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 100 == 0:
            size = len(dataloader.dataset)  # type: ignore
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,  # type: ignore
    epochs: int,
):
    """
    Train a model for a number of epochs

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing training data
    model : nn.Module
        Model to train
    loss_fn : nn.Module
        Loss function
    optimiser : torch.optim.Optimizer
        Optimiser used to update model parameters
    epochs : int
        Number of epochs to train for
    """

    for t in range(epochs):
        logger.info(f"Epoch {t+1}\n-------------------------------")
        epoch(dataloader, model, loss_fn, optimiser)
        test(dataloader, model, loss_fn)
    logger.info("Done!")


def test(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    device = get_device()
    size = len(dataloader.dataset)  # type: ignore
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def save_model(model: nn.Module, path: str = "./checkpoints/model.pth"):
    torch.save(model.state_dict(), path)


def load_model(
    model_fn: Callable[..., nn.Module], path: str = "./checkpoints/model.pth"
):
    model = model_fn().to(get_device())
    model.load_state_dict(torch.load(path))
    return model
