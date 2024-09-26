import torch
import toolz as tz
from toolz import curried
import polars as pl
import numpy as np
from itertools import tee
import os
from datetime import datetime
import sys

sys.path.append("..")
sys.path.append(".")
from kornia.augmentation import RandomAffine3D
from scripts.__helpful_parser import HelpfulParser
import torch
from uncertainty.training import load_checkpoint, checkpoint_dir_type, torchio_augmentation
from uncertainty.evaluation import (
    get_inference_mode,
    evaluate_prediction,
    evaluate_predictions,
    probability_map,
    entropy_map,
    variance_map
)
from uncertainty.data.h5 import save_pred_to_h5, load_pred_from_h5, load_xy_from_h5
from loguru import logger


def parameterised_inference(mode: str, n_outputs: int, model, config: dict):
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
        lambda inference: lambda x: inference(
            model=model,
            x=x,
            patch_size=config["patch_size"],
            subdivisions=2,
            batch_size=config["batch_size_eval"],
            output_channels=config["n_kernels_last"],
            #prog_bar=mode != "single",
            prog_bar=True
        ),
    )

def dump_aggregated_maps(i_preds_y):
    idx, preds_y = i_preds_y
    preds, y = list(preds_y[0]), preds_y[1]

    unique_folder_name = str(idx)
    maps_folder = os.path.join("prediction_maps", unique_folder_name)
    os.makedirs(maps_folder, exist_ok=True)

    # Create filenames for the maps
    prob_map_filename = os.path.join(maps_folder, "probability_map.pt")
    entropy_map_filename = os.path.join(maps_folder, "entropy_map.pt")
    variance_map_filename = os.path.join(maps_folder, "variance_map.pt")

    torch.save(probability_map(preds), prob_map_filename)
    torch.save(entropy_map(preds), entropy_map_filename)
    torch.save(variance_map(preds), variance_map_filename)

    return preds, y

def main(
    checkpoint_dir: str,
    metric_names: list[str],
    mode: str,
    average: str,
    class_names: list[str] | None,
    out_path: str,
    n_outputs: int,
    dataset_path: str
):

    model, config, indices, _, val = load_checkpoint(checkpoint_dir)
    model.eval()

    torch.set_grad_enabled(False)

    evaluation = (
        evaluate_prediction(average=average)
        if mode == "single"
        else evaluate_predictions(average=average)
    )
    inference = parameterised_inference(mode, n_outputs, model, config)

    if average == "none":
        assert (
            class_names is not None
        ), "You must specify class names when average == 'none'!"
    col_names = (
        metric_names
        if average != "none"
        # [class1_metric1, class2_metric1, ..., class1_metric2, class2_metric2, ...]
        else [f"{name}_{metric}" for metric in metric_names for name in class_names]
    )
    col_names_sorted = (
        metric_names
        if average != "none"
        # [class1_metric1, class1_metric2, ..., class2_metric1, class2_metric2, ...]
        else [f"{name}_{metric}" for name in class_names for metric in metric_names]
    )

    from itertools import islice
    tz.pipe(
        islice(load_xy_from_h5(dataset_path), 1),
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
        curried.map(dump_aggregated_maps) if mode != "single" else tz.identity,
        curried.map(lambda y_pred: evaluation(y_pred[0], y_pred[1], metric_names)),
        curried.map(lambda tensor: tensor.tolist()),
        lambda it: pl.LazyFrame(it, schema=col_names),
        lambda it: it.select(pl.col(col_names_sorted)),  # reorder columns
        lambda lf: lf.sink_csv(out_path),
    )


if __name__ == "__main__":
    parser = HelpfulParser(
        description="Load the model(s) from checkpoint(s), make inference on a dataset and output the metrics into a CSV file."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help='Path to the checkpoint directory containing "latest.ckpt", "indices.pt", and "config.pkl" or a directory containing these checkpoint directories with the same structure.',
        default="./checkpoints/unet_single"
    )
    # TODO: add option for test set too
    parser.add_argument(
        "--metrics",
        type=str,
        help="List of metric names separated by comma, can be any of 'dice', 'hd', 'hd95', 'asd', 'assd', 'recall', 'sen', 'precision', 'ppv' or 'surface_dice_[float]' where [float] is the threshold to use for surface Dice.",
        default="dice,surface_dice_1.5,hd95,assd,sen,ppv",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Inference mode, any of 'single', 'mcdo', 'tta', or 'ensemble'. Note that for 'ensemble', --checkpoint_dir MUST point to a directory containing directories of model checkpoints.",
        default="tta",
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
        default='prostate,bladder,rectum',
    )
    parser.add_argument(
        "--metric-out", type=str, help="Name of the output CSV file", default="eval.csv"
    )
    parser.add_argument(
        "--n-outputs",
        type=int,
        help="Number of outputs to predict for a single instance; ignored if mode is 'single' or 'ensemble'.",
        default=2,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the h5 dataset to perform inference on.",
        default="./test.h5"
    )

    args = parser.parse_args()
    detected_dir_type = checkpoint_dir_type(args.checkpoint_dir)
    if (detected_dir_type != "multiple" and args.mode in ["ensemble"]) or (detected_dir_type != "single" and args.mode in ["tta", "mcdo", "single"]):
        logger.critical("Invalid checkpoint path for the given inference mode.")
        exit(1)

    main(
        args.checkpoint_dir,
        args.metrics.split(","),
        args.mode,
        args.average,
        (
            args.class_names.split(",")
            if args.class_names is not None
            else args.class_names
        ),
        args.metric_out,
        args.n_outputs,
        args.dataset_path
    )
