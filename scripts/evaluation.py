from context import uncertainty as un
import os

import dill
from scripts.__helpful_parser import HelpfulParser


"""
input
    path to checkpoints
    best_model_name
    path to one dataset
    list of models
        - model (best)
        - ensemble
        - tta (best)
        - mc dropout (best)
    list of metrics
        - dice
        - ...

for each model...
    1. load model
    2. feed EACH instance into model and get patched output - inference
        3. calculate metric for each organ
            -> DSC [organ1, organ2, organ3]
            -> HD [organ1, organ2, organ3]
        4. get average organ-wise
            e.g. model1: [0.8, 0.7, 0.6]
                 model2: [0.7, 0.6, 0.5]
                 model3: [0.6, 0.5, 0.4]
                 torch.mean
    4. lazyframe: model, metrics


output
    csv file
    | model | organ_metrics...
"""


def main():
    pass


if __name__ == "__main__":
    config = un.config.configuration()
    parser = HelpfulParser(
        description="Train a model on a dataset of DICOM files or h5 files."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the dataset.h5 file containing list of (x, y) pairs.",
        default=os.path.join(config["staging_dir"], config["staging_fname"]),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path a folder containing model checkpoint latest.pl, train-validation split indices indices.pt, and the model configuration object config.pkl.",
        default=config["model_checkpoint_path"],
    )

    args = parser.parse_args()
    config["staging_dir"] = os.path.dirname(args.data_path)
    config["staging_fname"] = os.path.basename(args.data_path)
    checkpoint_path = args.checkpoint_path
    if args.retrain:
        with open(os.path.join(checkpoint_path, "config.pkl"), "rb") as f:
            config = dill.load(f)

    main(
        config,
        checkpoint_path,
        args.retrain,
        deep_supervision=not args.no_deep_supervision,
    )
