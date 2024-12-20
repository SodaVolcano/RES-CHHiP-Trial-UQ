import gc
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import polars as pl
import toolz as tz
import torch
from kornia.augmentation import RandomAffine3D
from loguru import logger
from toolz import curried


sys.path.append("..")
sys.path.append(".")
from scripts.__helpful_parser import HelpfulParser
from uncertainty.config import configuration
from uncertainty.data import load_scans_from_h5, torchio_augmentations
from uncertainty.data.h5 import save_prediction_to_h5, save_predictions_to_h5
from uncertainty.evaluation import (
    evaluate_prediction,
    evaluate_predictions,
    get_inference_mode,
)
from uncertainty.training import (
    LitModel,
    load_training_dir,
    select_ensembles,
    select_single_models,
)
from uncertainty.utils import (
    config_logger,
    side_effect,
    star,
    transform_nth,
    list_files,
    curry,
)


def get_parameterised_inference(
    mode: str,
    n_outputs: int,
    model: LitModel | list[LitModel],
    patch_size: int | tuple[int, int, int],
    batch_size: int,
    output_channels: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get unary inference function f(x) -> y_pred with parameters filled in
    """
    aug = torchio_augmentations()
    aug_batch = RandomAffine3D(
        5, align_corners=True, shears=0, scale=(0.9, 1.1), p=0.15
    )
    parameters = tz.pipe(
        {
            "patch_size": patch_size,
            "subdivisions": 2,
            "batch_size": batch_size,
            "output_channels": output_channels,
            "prog_bar": True,
        },
        (
            curried.assoc(key="n_outputs", value=n_outputs)
            if mode not in ["single", "ensemble"]
            else tz.identity
        ),
        curried.assoc(key="aug", value=aug) if mode == "tta" else tz.identity,
        (
            curried.assoc(key="batch_affine", value=aug_batch)
            if mode == "tta"
            else tz.identity
        ),
        curried.assoc(key="model" if mode != "ensemble" else "models", value=model),
    )

    return tz.pipe(
        mode,
        get_inference_mode,
        lambda inference: inference(**parameters),
    )  # type: ignore


def _to_tensors_if_numpy(
    arrays: tuple[np.ndarray | torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    """Convert x, y pair to torch tensor if they are numpy arrays"""
    return tz.pipe(
        arrays,
        curried.map(
            lambda arr: (
                torch.tensor(arr, requires_grad=False)
                if isinstance(arr, np.ndarray)
                else arr
            )
        ),
        tuple,
    )  # type: ignore


def perform_inference(
    h5_path: str,
    test_indices: list[str],
    inference_fn: Callable[[torch.Tensor], torch.Tensor],
    save_pred: Callable,
    save_pred_path: str,
    evaluate_pred: Callable,
) -> Iterable[list[float]]:
    """
    Return iterator of list of [patient_id, metric1, metric2, ...] for each prediction
    """
    save_pred = save_pred(save_pred_path)  # set h5_path

    return tz.pipe(
        load_scans_from_h5(h5_path, test_indices),
        # tuple format matches save_pred's func signature
        curried.map(
            lambda scan: (
                scan["patient_id"],
                None,  # don't save x, save space on disk
                scan["masks"],
                scan["volume"],
            )
        ),
        curried.map(_to_tensors_if_numpy),
        curried.map(transform_nth(3, inference_fn)),
        curried.map(tuple),
        curried.map(side_effect(save_pred)),
        curried.map(
            star(
                lambda patient_id, _, y, y_pred: (
                    [patient_id],
                    evaluate_pred(y_pred, y),
                )
            )
        ),
        curried.map(transform_nth(1, lambda tensor: tensor.tolist())),
        curried.map(tz.concat),
        curried.map(side_effect(lambda: gc.collect())),
    )


def to_csv(
    it: Iterable[list[float]],
    col_names: list[str],
    col_names_reordered: list[str],
    out_path: str | Path,
):
    """
    Given an iterator of metrics each starting with ID, compute metric-wise average and save to CSV
    """
    avg_metrics = tz.pipe(
        col_names_reordered[1:],  # skip patient_id
        curried.groupby(lambda col_name: col_name.split("_")[-1]),
        lambda groups: {
            f"avg_{metric}": pl.mean_horizontal(*cols)
            for metric, cols in groups.items()
        },
    )

    tz.pipe(
        pl.LazyFrame(it, schema=col_names),
        lambda lf: lf.select(pl.col(col_names_reordered)),
        # add average metrics onto the end
        lambda lf: lf.with_columns(**avg_metrics),
        lambda lf: lf.sink_csv(out_path),
    )


def compute_fold_avg_csv(csv_dir: Path, col_names: list[str]):
    """
    Compute the average of the CSV files in the directory and save to a new CSV file
    """
    tz.pipe(
        csv_dir,
        list_files,
        curried.map(pl.scan_csv),
        curry(pl.concat)(how="vertical"),
        lambda lf: lf.group_by("patient_id"),
        lambda lf: lf.agg(
            [pl.col(col).mean() for col in col_names[1:]]  # [1:] skip patient_id
        ),
        lambda lf: lf.sink(csv_dir / "fold_avg.csv"),
    )


@torch.no_grad()
def infer_using_mode(
    h5_path: str,
    ckpt_dict: dict[str, LitModel],
    mode: str,
    n_outputs: int,
    patch_size: int | tuple[int, int, int],
    batch_size: int,
    output_channels: int,
    metric_names: list[str],
    class_names: list[str],
    test_indices: list[str],
    pred_dir: Path,
    csv_dir: Path,
):
    mode_specific_fns = {
        "single": (select_single_models, save_prediction_to_h5, evaluate_prediction),
        "mcdo": (select_single_models, save_prediction_to_h5, evaluate_prediction),
        "tta": (select_single_models, save_prediction_to_h5, evaluate_prediction),
        "ensemble": (select_ensembles, save_predictions_to_h5, evaluate_predictions),
    }
    select_model = mode_specific_fns[mode][0]
    save_pred = mode_specific_fns[mode][1]
    evaluator = mode_specific_fns[mode][2](metric_names=metric_names, average="none")

    # class1_metric1, class2_metric1, ..., class1_metric2, class2_metric2, ...
    col_names = ["patient_id"] + [
        f"{cls_name}_{metric}" for metric in metric_names for cls_name in class_names
    ]
    # class1_metric1, class1_metric2, ..., class2_metric1, class2_metric2, ...
    col_names_reordered = ["patient_id"] + [
        f"{cls_name}_{metric}" for cls_name in class_names for metric in metric_names
    ]

    model_dict = select_model(ckpt_dict)
    for model_name, model in model_dict.items():
        inference_fn = get_parameterised_inference(
            mode, n_outputs, model, patch_size, batch_size, output_channels
        )
        eval_metrics_it = perform_inference(
            h5_path,
            test_indices,
            inference_fn,
            save_pred,
            str(pred_dir / f"{mode}_{model_name}.h5"),
            evaluator,
        )
        to_csv(
            eval_metrics_it,
            col_names,
            col_names_reordered,
            csv_dir / f"{mode}_{model_name}.csv",
        )

    compute_fold_avg_csv(csv_dir, col_names_reordered)


def main(
    train_dir: str,
    metrics: list[str],
    modes: list[str],
    class_names: list,
    n_outputs: int,
    h5_path: str,
    pred_dir: Path,
    csv_dir: Path,
):
    config, _, (_, test_indices), fold_dict = load_training_dir(train_dir)

    for fold_name, ckpt_dict in fold_dict.items():
        for mode in modes:
            infer_using_mode(
                h5_path,
                ckpt_dict,
                mode,
                n_outputs,
                config["data__patch_size"],
                config["training__batch_size"],
                config["model__output_channels"],
                metrics,
                class_names,
                test_indices,  # type: ignore
                pred_dir / f"fold-{fold_name}",
                csv_dir / f"fold-{fold_name}",
            )


if __name__ == "__main__":
    parser = HelpfulParser(
        description="Load the model(s) from checkpoint(s), make inference on a dataset and output the metrics into a CSV file."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to the configuration file.",
        default="configuration.yaml",
    )
    parser.add_argument(
        "--train-dir",
        "-i",
        type=str,
        help="Path to the training directory containing folders for each fold, with each folder containing directories of model checkpoints.",
        required=False,
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="List of metric names separated by comma, can be any of 'dice', 'hd', 'hd95', 'asd', 'assd', 'recall', 'sen', 'precision', 'ppv', 'surface_dice', 'mean_variance', 'mean_entropy', or 'surface_dice_[float]' where [float] is the threshold to use for surface Dice.",
        required=False,
    )
    parser.add_argument(
        "--modes",
        type=str,
        help="List of comma-separated inference modes, any of 'single', 'mcdo', 'tta', or 'ensemble'.",
        required=False,
    )
    parser.add_argument(
        "--class_names",
        type=str,
        help="List of names separated by comma for each class to predict.",
        required=False,
    )
    parser.add_argument(
        "--n-outputs",
        type=int,
        help="Number of outputs to predict for a single instance; ignored if mode is 'single' or 'ensemble'.",
        required=False,
    )
    parser.add_argument(
        "--h5-path",
        "-d",
        type=str,
        help="Path to the H5 dataset of PatientScan dictionaries.",
        required=False,
    )
    parser.add_argument(
        "--pred-dir",
        "-p",
        type=str,
        help="Path to the directory where predictions will be saved. If not provided, the pred_dir from the configuration file will be used.",
        required=False,
    )
    parser.add_argument(
        "--csv-dir",
        "-o",
        type=str,
        help="Path to the directory where the CSV files will be saved. If not provided, the csv_dir from the configuration file will be used.",
    )

    args = parser.parse_args()
    config = configuration(args.config)
    if args.logging:
        logger.enable("uncertainty")
        config_logger(**config)

    main(
        args.train_dir or config["training__train_dir"],
        args.metrics.split(",") if args.metrics else config["evaluation__metrics"],
        args.modes.split(",") if args.modes else config["evaluation__modes"],
        (
            args.class_names.split(",")
            if args.class_names
            else config["evaluation__class_names"]
        ),
        args.n_outputs or config["evaluation__n_outputs"],
        args.h5_path or config["data__h5_path"],
        Path(args.pred_dir or config["evaluation__predictions_dir"]),
        Path(args.csv_dir or config["evaluation__csv_dir"]),
    )
