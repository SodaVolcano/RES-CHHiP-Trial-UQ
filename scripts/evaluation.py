import gc
import os
import sys
from typing import Callable

import numpy as np
import polars as pl
import toolz as tz
import torch
from toolz import curried

sys.path.append("..")
sys.path.append(".")
import h5py
import torch
from kornia.augmentation import RandomAffine3D
from loguru import logger
from tqdm import tqdm

from scripts.__helpful_parser import HelpfulParser
from uncertainty.data.h5 import load_xy_from_h5
from uncertainty.data.processing import crop_to_body
from uncertainty.evaluation import (
    entropy_map,
    evaluate_prediction,
    evaluate_predictions,
    get_inference_mode,
    probability_map,
    variance_map,
)
from uncertainty.training import (
    checkpoint_dir_type,
    load_checkpoint,
    torchio_augmentation,
)
from uncertainty.utils.wrappers import curry


def parameterised_inference(
    mode: str, n_outputs: int, model, config: dict
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get unary inference function f(x) -> y_pred with parameters filled in
    """
    aug = torchio_augmentation()
    aug_batch = RandomAffine3D(
        5, align_corners=True, shears=0, scale=(0.9, 1.1), p=0.15
    )
    return tz.pipe(
        mode,
        get_inference_mode,
        # specify how many outputs to produce if required
        lambda inference: (
            inference(n_outputs=n_outputs)
            if mode not in ["single", "ensemble"]
            else inference
        ),
        lambda inference: (
            inference(aug=aug, batch_affine=aug_batch) if mode == "tta" else inference
        ),
        (
            (lambda inference: inference(model=model))
            if mode != "ensemble"
            else (lambda inference: inference(models=model))
        ),
        lambda inference: lambda x: inference(
            x=x,
            patch_size=config["patch_size"],
            subdivisions=2,
            batch_size=config["batch_size_eval"],
            output_channels=config["n_kernels_last"],
            # prog_bar=mode != "single",
            prog_bar=True,
        ),
    )  # type: ignore


@curry
def dump_pred(i_pred_y):
    idx, pred_y = i_pred_y
    pred, y = pred_y[0], pred_y[1]

    unique_folder_name = str(idx)
    maps_folder = os.path.join("single_preds", unique_folder_name)
    os.makedirs(maps_folder, exist_ok=True)

    torch.save(y, os.path.join(maps_folder, "label.pt"))
    torch.save(pred, os.path.join(maps_folder, "pred.pt"))
    return pred, y


@curry
def dump_aggregated_maps(i_preds_y, map_folder_path, class_names: list[str]):
    idx, preds_y = i_preds_y
    preds, y = list(preds_y[0]), preds_y[1]

    unique_folder_name = str(idx)
    maps_folder = os.path.join(map_folder_path, unique_folder_name)
    os.makedirs(maps_folder, exist_ok=True)

    preds_filename = "./preds"
    prob_map_filename = os.path.join(maps_folder, "probability_map")
    entropy_map_filename = os.path.join(maps_folder, "entropy_map")
    variance_map_filename = os.path.join(maps_folder, "variance_map")

    with h5py.File(os.path.join(maps_folder, "data.h5"), "w") as hf:
        hf.create_dataset("y", data=y, compression="gzip")
        group = hf.create_group("preds")
        for i, pred in enumerate(preds):
            group.create_dataset(str(i), data=pred, compression="gzip")

    # Save predictions for each class to calculate maps per-class later (avoid overflowing RAM)
    for class_idx in range(len(class_names)):
        torch.save(
            [pred[class_idx] for pred in preds],
            f"{preds_filename}_{class_names[class_idx]}.pt",
        )

    for pred_fname, class_name in [
        (f"{preds_filename}_{name}.pt", name) for name in class_names
    ]:
        # Load the saved preds with memory mapping
        class_preds = torch.load(pred_fname, mmap=True, weights_only=True)

        torch.save(probability_map(class_preds), f"{prob_map_filename}_{class_name}.pt")
        torch.save(entropy_map(class_preds), f"{entropy_map_filename}_{class_name}.pt")
        torch.save(
            variance_map(class_preds), f"{variance_map_filename}_{class_name}.pt"
        )

    return preds, y


@torch.no_grad()
def perform_inference(
    checkpoint_dir: str,
    metric_names: list[str],
    mode: str,
    average: str,
    class_names: list[str] | None,
    out_path: str,
    n_outputs: int,
    dataset_path: str,
    map_path: str,
):
    detected_dir_type = checkpoint_dir_type(checkpoint_dir)
    if (detected_dir_type != "multiple" and mode in ["ensemble"]) or (
        detected_dir_type != "single" and mode in ["tta", "mcdo", "single"]
    ):
        logger.critical("Invalid checkpoint path for the given inference mode.")
        exit(1)

    if detected_dir_type == "single":
        model, config, _, _, _ = load_checkpoint(checkpoint_dir)
        model.eval()
    else:
        model = []
        config = None
        for ckpt in os.listdir(checkpoint_dir):
            one_model, config, _, _, _ = load_checkpoint(
                os.path.join(checkpoint_dir, ckpt)
            )
            one_model.eval()
            model.append(one_model)

    torch.set_grad_enabled(False)

    evaluation = (
        evaluate_prediction(average=average)
        if mode == "single"
        else evaluate_predictions(average=average)
    )
    inference = parameterised_inference(mode, n_outputs, model, config)  # type: ignore

    if average == "none":
        assert (
            class_names is not None
        ), "You must specify class names when average == 'none'!"
    col_names = (
        metric_names
        if average != "none"
        # [class1_metric1, class2_metric1, ..., class1_metric2, class2_metric2, ...]
        else [f"{name}_{metric}" for metric in metric_names for name in class_names]  # type: ignore
    )
    col_names_sorted = (
        metric_names
        if average != "none"
        # [class1_metric1, class1_metric2, ..., class2_metric1, class2_metric2, ...]
        else [f"{name}_{metric}" for name in class_names for metric in metric_names]  # type: ignore
    )

    def collect_garbage(x):
        # Memory aren't cleared after each iteration for some reason so we do it manually...
        gc.collect()
        return x

    tz.pipe(
        load_xy_from_h5(dataset_path),
        curried.map(
            lambda xy: crop_to_body(xy[0], xy[1])
        ),  # reduce storage space of array
        # to torch tensor if numpy
        curried.map(
            lambda xy: (
                tuple(map(torch.tensor, xy))
                if isinstance(xy[0], np.ndarray) or isinstance(xy[1], np.ndarray)
                else xy
            )
        ),
        curried.map(lambda xy: (inference(xy[0]), xy[1])),
        enumerate,
        curried.map(
            dump_aggregated_maps(map_folder_path=map_path, class_names=class_names)
            if mode != "single"
            else dump_pred
        ),
        curried.map(lambda y_pred: evaluation(y_pred[0], y_pred[1], metric_names)),
        curried.map(lambda tensor: tensor.tolist()),
        curried.map(collect_garbage),
        lambda it: pl.LazyFrame(it, schema=col_names),
        lambda it: it.select(pl.col(col_names_sorted)),  # reorder columns
        lambda lf: lf.sink_csv(out_path),
    )


@torch.no_grad()
def main(
    single_model_dir: str,
    ensemble_dir: str,
    inference_modes: list[str],
    metric_names: list[str],
    average: str,
    class_names: list[str] | None,
    out_path: str,
    n_outputs: int,
    dataset_path: str,
):

    for mode in inference_modes:
        checkpoint_dir = ensemble_dir if mode == "ensemble" else single_model_dir
        perform_inference(
            checkpoint_dir,
            list(
                filter(
                    lambda name: (
                        name not in ["mean_variance", "mean_entropy", "pairwise_dice"]
                        if mode == "single"
                        else True
                    ),
                    metric_names,
                )
            ),
            mode,
            average,
            class_names,
            f"{mode}_{out_path}",
            n_outputs,
            dataset_path,
            map_path=f"prediction_maps/{mode}",
        )


if __name__ == "__main__":
    parser = HelpfulParser(
        description="Load the model(s) from checkpoint(s), make inference on a dataset and output the metrics into a CSV file."
    )
    parser.add_argument(
        "--ensemble_dir",
        type=str,
        help='Path to a directory containing checkpoint directories each containing "latest.ckpt", "indices.pt", and "config.pkl". Not required if --modes does not contain "ensemble".',
        default="./checkpoints/",
    )
    parser.add_argument(
        "--single_model_dir",
        type=str,
        help='Path to the checkpoint directory containing "latest.ckpt", "indices.pt", and "config.pkl" for a single model. Not required if --modes does not contain "single", "mcdo", and "tta".',
        default="./unet_single/",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="List of metric names separated by comma, can be any of 'dice', 'hd', 'hd95', 'asd', 'assd', 'recall', 'sen', 'precision', 'ppv', 'surface_dice', 'mean_variance', 'mean_entropy', or 'surface_dice_[float]' where [float] is the threshold to use for surface Dice.",
        default="dice,surface_dice_1.0,surface_dice_1.5,surface_dice_2.0,surface_dice_2.5,surface_dice_3.0,surface_dice_5.099,surface_dice_13.928,hd95,assd,sen,ppv,pairwise_dice",
    )
    parser.add_argument(
        "--modes",
        type=str,
        help="List of comma-separated inference modes, any of 'single', 'mcdo', 'tta', or 'ensemble'.",
        default="mcdo,tta,ensemble",
    )
    parser.add_argument(
        "--average",
        type=str,
        help="Average mode when calculating metrics for multiple classes. Any of 'micro', 'macro', or 'none' to disable averaging across classes. If 'none', --class_names must be specified.",
        default="none",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        help="List of names separated by comma for each class to predict, only used if --average=none and is ignored if otherwise.",
        default="prostate,bladder,rectum",
    )
    parser.add_argument(
        "--metric-out",
        type=str,
        help="Base name of the output CSV file",
        default="eval.csv",
    )
    parser.add_argument(
        "--n-outputs",
        type=int,
        help="Number of outputs to predict for a single instance; ignored if mode is 'single' or 'ensemble'.",
        default=20,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the h5 dataset to perform inference on.",
        default="./test.h5",
    )

    args = parser.parse_args()

    main(
        args.single_model_dir,
        args.ensemble_dir,
        args.modes.split(","),
        args.metrics.split(","),
        args.average,
        (
            args.class_names.split(",")
            if args.class_names is not None
            else args.class_names
        ),
        args.metric_out,
        args.n_outputs,
        args.dataset_path,
    )
