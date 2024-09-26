import torch
import toolz as tz
from toolz import curried
import polars as pl
import numpy as np
from itertools import tee

import sys

sys.path.append("..")
sys.path.append(".")
from kornia.augmentation import RandomAffine3D
from scripts.__helpful_parser import HelpfulParser
from uncertainty.training.augmentations import torchio_augmentation
import torch
from uncertainty.training.checkpoint import load_checkpoint
from uncertainty.evaluation import (
    sliding_inference,
    tta_inference,
    ensemble_inference,
    mc_dropout_inference,
    evaluate_prediction,
    evaluate_predictions,
)
from uncertainty.data.h5 import save_pred_to_h5, load_pred_from_h5


def _get_inference_mode(mode: str):
    return {
        "single": sliding_inference,
        "tta": tta_inference,
        "ensemble": ensemble_inference,
        "mcdo": mc_dropout_inference,
    }[mode]


def main(
    checkpoint_dir: str,
    metric_names: list[str],
    mode: str,
    average: str,
    class_names: list[str] | None,
    out_path: str,
    dump_results: bool,
    pred_out_path: str,
    n_outputs: int,
):

    model, config, indices, _, val = load_checkpoint(checkpoint_dir)
    model.eval()
    aug = torchio_augmentation()
    aug_batch = RandomAffine3D(
        5, align_corners=True, shears=0, scale=(0.9, 1.1), p=0.15
    )
    torch.set_grad_enabled(False)

    inference = tz.pipe(
        mode,
        _get_inference_mode,
        # parameterise the inference function to get a unary function (x) -> y_pred
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
            prog_bar=mode != "single",
        ),
    )
    evaluation = (
        evaluate_prediction(average=average)
        if mode == "single"
        else evaluate_predictions(average=average)
    )

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

    def map_to_numpy(input_):
        if isinstance(input_, np.ndarray):
            return torch.from_numpy(input_)
        elif isinstance(input_, torch.Tensor):
            return input_
        else:
            return map(lambda x: torch.from_numpy(x), input_)

    def dump_results_to_h5(pred_ys):
        it1, it2 = tee(pred_ys, 2)
        pred, ys = map(curried.get(0), it1), map(curried.get(1), it2)
        save_pred_to_h5(pred, indices["val_indices"], path=".", name=pred_out_path)
        return tz.pipe(
            zip(
                load_pred_from_h5(pred_out_path, map(str, indices["val_indices"])),
                ys,
            ),
            # to torch if is numpy
            curried.map(
                lambda pred_y: tuple(
                    map(
                        map_to_numpy,
                        pred_y,
                    )
                )
            ),
        )

    tz.pipe(
        val,
        curried.map(
            lambda xy: (
                tuple(map(torch.tensor, xy))
                if isinstance(xy[0], np.ndarray) or isinstance(xy[1], np.ndarray)
                else xy
            )
        ),
        curried.map(lambda xy: (inference(xy[0]), xy[1])),
        dump_results_to_h5 if dump_results else tz.identity,
        curried.map(lambda y_pred: evaluation(y_pred[0], y_pred[1], metric_names)),
        curried.map(lambda tensor: tensor.tolist()),
        lambda results: zip(indices["val_indices"], results),
        # concat index with the metrics into a single list for polars
        curried.map(lambda idx_res: [idx_res[0]] + idx_res[1]),
        lambda it: pl.LazyFrame(it, schema=["val_idx"] + col_names),
        lambda it: it.select(pl.col(["val_idx"] + col_names_sorted)),  # reorder columns
        lambda lf: lf.sink_csv(out_path),
    )


if __name__ == "__main__":
    parser = HelpfulParser(
        description="Load the model(s) from checkpoint(s), make inference on a dataset and output the metrics into a CSV file."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help='Path to the checkpoint directory containing "last.pkl", "indices.pt", and "config.pkl" or a directory containing these checkpoint directories with the same structure.',
    )
    # TODO: add option for test set too
    parser.add_argument(
        "--metrics",
        type=str,
        help="List of metric names separated by comma, can be any of 'dice', 'hd', 'hd95', 'asd', 'assd', 'recall', 'sen', 'precision', 'ppv' or 'surface_dice_[float]' where [float] is the threshold to use for surface Dice.",
        default="dice,surface_dice_0.5,surface_dice_1.0,surface_dice_1.5,surface_dice_2.0,surface_dice_2.5,surface_dice_3.0,hd95,asd,assd,sen,ppv",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Inference mode, any of 'single', 'mcdo', 'tta', or 'ensemble'. Note that for 'ensemble', --checkpoint_dir MUST point to a directory containing directories of model checkpoints.",
        default="single",
    )
    parser.add_argument(
        "--average",
        type=str,
        help="Average mode when calculating metrics for multiple classes. Any of 'micro', 'macro', or 'none' to disable averaging across classes. If 'none', --class_names must be specified.",
        default="macro",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        help="List of names separated by comma for each class to predict, only used if --average=none and is ignored if otherwise.",
        default=None,
    )
    parser.add_argument(
        "--metric-out", type=str, help="Name of the output CSV file", default="eval.csv"
    )
    parser.add_argument(
        "--dump-results",
        action="store_true",
        help="Whether to store all model predictions after inference; if set the predictions are saved to pred.h5",
        default=False,
    )
    parser.add_argument(
        "--pred-out",
        type=str,
        help="Name of the H5 file to store (x, y, y_pred) if --dump-results is specified",
        default="predictions.h5",
    )
    parser.add_argument(
        "--n-outputs",
        type=int,
        help="Number of outputs to predict for a single instance; ignored if mode is 'single' or 'ensemble'.",
        default=50,
    )

    args = parser.parse_args()

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
        args.dump_results,
        args.pred_out,
        args.n_outputs,
    )
