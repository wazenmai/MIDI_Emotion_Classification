from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.model.net import SAN
from src.task.pipeline import PEmoPipeline
from src.task.runner import Runner


def get_config(args: Namespace) -> DictConfig:
    parent_config_dir = Path("conf")
    child_config_dir = parent_config_dir / args.dataset
    model_config_dir = child_config_dir / "model"
    pipeline_config_dir = child_config_dir / "pipeline"
    runner_config_dir = child_config_dir / "runner"

    config = OmegaConf.create()
    model_config = OmegaConf.load(model_config_dir / f"{args.model}.yaml")
    pipeline_config = OmegaConf.load(pipeline_config_dir / f"{args.pipeline}.yaml")
    runner_config = OmegaConf.load(runner_config_dir / f"{args.runner}.yaml")
    config.update(model=model_config, pipeline=pipeline_config, runner=runner_config)
    return config


def get_tensorboard_logger(args: Namespace) -> TensorBoardLogger:
    logger = TensorBoardLogger(
        save_dir=f"exp/{args.dataset}", name=args.model, version=f"{args.pipeline}_{args.runner}/"
    )
    return logger


def get_checkpoint_callback(args: Namespace) -> ModelCheckpoint:
    prefix = f"exp/{args.dataset}/{args.model}/{args.pipeline}_{args.runner}/"
    suffix = "{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}"
    filepath = prefix + suffix
    checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        save_top_k=1,
        monitor="val_loss",
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback


def get_early_stop_callback(args: Namespace) -> EarlyStopping:
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min"
    )
    return early_stop_callback


def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    config = get_config(args)
    logger = get_tensorboard_logger(args)
    checkpoint_callback = get_checkpoint_callback(args)
    early_stop_callback = get_early_stop_callback(args)

    pipeline = PEmoPipeline(pipline_config=config.pipeline, model_config=config.model)
    model = SAN(**config.model.params)
    runner = Runner(model, config.runner)

    trainer = Trainer(
        **config.runner.trainer.params,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback],
    )
    trainer.fit(runner, datamodule=pipeline)

    result = trainer.test(runner, datamodule=pipeline)
    with open(f"exp/{args.dataset}/{args.model}/{args.pipeline}_{args.runner}/results.json", mode="w") as io:
        json.dump(result, io, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model",default="SAN_Magenta_2Class_head1",type=str)
    parser.add_argument("--dataset", default="PEmo", type=str)
    parser.add_argument("--pipeline", default="v_magenta", type=str)
    parser.add_argument("--runner", default="gpu0", type=str)
    parser.add_argument("--reproduce", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
 