"""
slop
"""

import sys

sys.path.append(".")
sys.path.append("..")
from itertools import combinations_with_replacement

import h5py
import numpy as np
import toolz as tz
import torch
from scipy import optimize
from toolz import curried

from uncertainty.evaluation import surface_dice
from uncertainty.utils.wrappers import curry


def pairwise_surface_dice(preds, thresh):
    return tz.pipe(
        preds,
        lambda preds: combinations_with_replacement(preds, 2),
        curried.map(lambda xx: (torch.from_numpy(xx[0]), torch.from_numpy(xx[1]))),
        curried.filter(lambda x: not x[0].equal(x[1])),
        curried.map(lambda x: surface_dice(x[0], x[1], thresh)),
        list,
        torch.stack,
        lambda preds: torch.mean(preds, dim=0),
    )


def avg_surface_dice(masks_lst, thresh):
    return torch.mean(
        torch.tensor([pairwise_surface_dice(masks, thresh) for masks in masks_lst])
    )


prostates = []
bladders = []
rectums = []
with h5py.File("./multilabel.h5", "r") as hf:
    for i in range(5):
        prostates.append(np.expand_dims(hf[str(i)]["y"]["prostate"][:].astype(bool), 1))
        bladders.append(np.expand_dims(hf[str(i)]["y"]["bladder"][:].astype(bool), 1))
        rectums.append(np.expand_dims(hf[str(i)]["y"]["rectum"][:].astype(bool), 1))


def find_thresh(dataset):
    @curry
    def objective_function(threshold):
        return avg_surface_dice(dataset, threshold) - 0.95

    result = optimize.root_scalar(objective_function, bracket=[0, 15], method="brentq")

    if result.converged:
        optimal_threshold = result.root
        print(f"Optimal threshold: {optimal_threshold}")
    else:
        print("No solution found.")


find_thresh(prostates)  # Optimal threshold: 5.09901951359262
find_thresh(bladders)  # Optimal threshold: 3.0000000000010094
find_thresh(rectums)  # Optimal threshold: 13.928388277184078
