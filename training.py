from argparse import ArgumentParser
from datetime import datetime

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import os

from cnn_model_builder import ConvModel
from clap_dataset import ClapDataModule

# cmd: python training.py --model mobilenet_v3 --epochs 30 --batch_size 64 --sliding_window_size 3.0 --gpus 0

if __name__ == "__main__":
    training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mobilenet_v3", help="Model architecture"
    )
    parser.add_argument("--pretrained", help="pretrained model", action="store_true")
    parser.add_argument("--ckpt", type=str, help="Model checkpoint path", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sliding_window_size", type=float, default=3)
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--gpus", nargs="+", help="List of GPU indices", type=int)

    args = parser.parse_args()

    model_config_file = open("configs/models.yaml", mode="r")
    model_cfg = yaml.load(model_config_file, Loader=yaml.FullLoader)

    data_config_file = open("configs/data.yaml", mode="r")
    data_cfg = yaml.load(data_config_file, Loader=yaml.FullLoader)

    exp_name = (
        f"{training_timestamp}-{args.model}-window{args.sliding_window_size:05.02f}sec"
    )

    wandb_logger = WandbLogger(project="ClapDetector", name=exp_name, log_model=False)

    # Load PyTorch Lightning trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{args.model}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename=exp_name,
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.gpus,
        logger=wandb_logger,
    )
    
    clap_datamodule = ClapDataModule(
        data_config=data_cfg,
        batch_size=args.batch_size,
        sliding_window_size=args.sliding_window_size,
    )

    if args.pretrained:
            model = ConvModel.load_from_checkpoint(args.ckpt)
            print(f"[INFO] Model Loaded from {args.ckpt}")
    else:
        model = ConvModel(
            model_architecture=args.model,
            **model_cfg[args.model],
        )

    # Training
    trainer.fit(model, datamodule=clap_datamodule)
    
    # Save the model
    model_weights_dir = "model_weights"
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)
    model_save_path = os.path.join(model_weights_dir, f"{exp_name}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Testing
    label_dict = clap_datamodule.get_label_dict()
    preds = trainer.predict(
        model=model, datamodule=clap_datamodule, return_predictions=True
    )

    # process_predictions(preds, exp_name, label_dict)