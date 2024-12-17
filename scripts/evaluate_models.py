from pathlib import Path
from typing import Callable, Iterable
import gc
import sys
import numpy as np
import polars as pl
import toolz as tz
import torch
from toolz import curried
from kornia.augmentation import RandomAffine3D
from loguru import logger


sys.path.append("..")
sys.path.append(".")
from uncertainty.data.h5 import save_prediction_to_h5, save_predictions_to_h5
from uncertainty.training import (
    LitModel,
    load_training_dir,
    select_ensembles,
    select_single_models,
)
from uncertainty.config import configuration
from uncertainty.data import torchio_augmentations, load_scans_from_h5
from uncertainty.evaluation import (
    get_inference_mode,
    evaluate_prediction,
    evaluate_predictions,
)
from uncertainty.utils import config_logger, transform_nth, side_effect, star

from scripts.__helpful_parser import HelpfulParser


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
    # load checkpoint

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


"""
1. load_training_dir to get
    {'fold_1': {'model1": ...}, ...}
2. for each fold,
    1. for each mode (single, mcdo, tta, ensemble)
        1. get inference function
        2. select relevant models -> select_single_models/ensembles()
        3. for each model in dict...
            1. perform_inference
            2. save CSV in top-most dir as fold-{idx}-mode-{mode}-model-{model}.csv
    
h5_path = pred_dir / f"fold-{idx}-mode-{mode}-model-{model}.h5"


evaluate for EACH FOLD
    for EACH MODE
        for EACH MODEL




input:
    {
        'model-1': <model>,
        'model-2': <model>,
        'model2-1': <model>,
        'model2-2': <model>,
    }
modes:
    - single/mcdo/tta
        1. select single model from potentially list of models
        2. inference - single/mcdo/tta

    - ensemble
        1. accumulate into list
           
        2. ensemble inference



old code pseudocode:
torch.set_grad_enabled(False)


FOR EACH MODE...

    DONE -------------------1. load model(s)
        - if single:
            - load the single model
            - model.eval()
        - else if ensemble:
            - load all models into LIST
            - model.eval() for each model

    don't need to change ----------- 2. 
        evaluation = (
            evaluate_prediction(average=average)
            if mode == "single"
            else evaluate_predictions(average=average)
        )

    DONE ------------- 3. get inference function
        - get aug and aug_batch
        - given inference mode... get_inference_mode()
        - if inference fn is tta or mcdo, set PARAMS n_outputs
        - if tta, set PARAMS aug and aug_batch
        - if ensemble, set PARAM models, else PARAM model
        - parameterise with rest of the params...

    4. organise column names for the CSV
        - !!!CHANGE: use 'none' AND 'macro' average modes
        - get list of column names as [f"{name}_{metric}" for metric in metric_names for name in class_names]
            class1_metric1, class2_metric1, ..., total_metric1
        - also, get a sorted version that swaps "for" order, as
            [f"{name}_{metric}" for name in class_names for metric in metric_names]


        grow_csv:
            INPUT:
                - col_names
                - iterator yielding rows

            1. given iterator, pl.LazyFrame(it, schema=col_names)
            2. compute average columns...



            
    5. perform inference
        1. load (x, y) pairs from h5
        2. if numpy, convert to torch tensor
        3. map tuple (inference(x), y)
        4. dump aggregated maps if not single, else dump single prediction
        5. evaluate y and y_pred using metric names
        6. convert to list
        7. side-effect: collect garbage
        8. create lazyframe with the iterator, pass in column names
        9. reorder columns using sorted column names
        10. sink to csv
    
    DONE - dump predictions
"""


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


def _to_tensors_if_np(
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
    Return iterator of list of metrics for each prediction
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
        curried.map(_to_tensors_if_np),
        curried.map(transform_nth(3, inference_fn)),
        curried.map(tuple),
        curried.map(side_effect(save_pred)),
        curried.map(star(lambda _, __, y, y_pred: evaluate_pred(y_pred, y))),
        curried.map(lambda tensor: tensor.tolist()),
        curried.map(side_effect(lambda: gc.collect())),
    )


def grow_csv():
    pass


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
    test_indices: list[str],
    pred_dir: Path,
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


def main(
    train_dir: str,
    metrics: list[str],
    modes: list[str],
    class_names: list,
    n_outputs: int,
    h5_path: str,
    pred_dir: Path,
):
    config, _, (_, test_idx), fold_dict = load_training_dir(train_dir)
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
                test_idx,  # type: ignore
                pred_dir / f"fold-{fold_name}",
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
        Path(args.pred_dir or config["evaluation__pred_dir"]),
    )
