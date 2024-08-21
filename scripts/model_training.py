from context import uncertainty as un
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from lightning import Trainer

from scripts.__helpful_parser import HelpfulParser
from uncertainty.config import Configuration


def name_to_model(name: str):
    return {
        "unet": un.models.unet.UNet,
        "mc_dropout_unet": un.models.MCDropoutUNet,
    }[name]


def main(
    config: Configuration, model_name: str, ensemble_size: int, checkpoint_path: str
):
    model_fn = name_to_model(model_name)

    if ensemble_size > 1:
        model = un.models.DeepEnsemble(model_fn, ensemble_size, config=config)
    else:
        model = model_fn(config=config)

    model = un.training.LitSegmentation(model, deep_supervision=True, config=config)
    data = un.training.SegmentationData(config)
    pbar = TQDMProgressBar()

    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        dirpath=f"{checkpoint_path}/model1",
        filename="{epoch:02d}-{val/loss:.2f}",
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor="val/loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )

    trainer = Trainer(max_epochs=10, callbacks=[checkpoint, early_stopping, pbar])
    trainer.fit(model, data)


if __name__ == "__main__":
    config = un.config.configuration()
    parser = HelpfulParser(
        description="Train a model on a dataset of DICOM files or h5 files."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the folder containing h5 files.",
        default=config["staging_dir"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["unet", "mc_dropout_unet"],
        help="Name of the model to train.",
        required=True,
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        help="Number of models in the ensemble. Default is 1 (no ensemble training).",
        default=1,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to save model checkpoints.",
        default=config["model_checkpoint_path"],
    )
    args = parser.parse_args()
    config["staging_dir"] = args.data_path

    main(config, args.model_name, args.ensemble_size, args.checkpoint_path)
