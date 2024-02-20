import glob
import os
import re
from argparse import ArgumentParser
import json
import pytorch_lightning as pl
import yaml
from clap_dataset import ClapDataset, ClapDataModule
from cnn_model_builder import ConvModel

if __name__ == '__main__':
    
    model_config_file = open('configs/models.yaml', mode='r')
    model_cfg = yaml.load(model_config_file, Loader=yaml.FullLoader)

    data_config_file = open('configs/data.yaml', mode='r')
    data_cfg = yaml.load(data_config_file, Loader=yaml.FullLoader)

    # Command Line Argument
    parser = ArgumentParser()
    parser.add_argument("--model", type=str,
                        default='mobilenet_v3', help="Model architecture")
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sliding_window_size", type=float, default=3)
    parser.add_argument('--gpus', nargs='+',
                        help='List of GPU indices', type=int)
    
    # cmd: python inference.py --model resnet18 --visualize --batch_size 64 --gpus 2 --ckpt ./checkpoints/resnet18/

    args = parser.parse_args()

    if args.ckpt == 'latest':
        ckpt_list = glob.glob(f"./checkpoints/{args.model}/*.ckpt")
        assert len(
            ckpt_list) > 0, 'No checkpoint found; Please train and save the model first'
        checkpoint_path = max(ckpt_list, key=os.path.getctime)
    else:
        checkpoint_path = args.ckpt

    # define experiment name
    exp_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    # colored text
    print('\x1b[6;30;42m' + f'[INFO] EXPERIMENT NAME: {exp_name}' + '\x1b[0m')


    model = ConvModel.load_from_checkpoint(args.ckpt)
    print(f'[INFO] Model Loaded from {args.ckpt}')


    clap_datamodule = ClapDataModule(
        data_config=data_cfg,
        batch_size=args.batch_size,
        sliding_window_size=args.sliding_window_size,
    )

    # model prediction
    trainer = pl.Trainer(accelerator='gpu', devices=args.gpus)

    preds = trainer.predict(
        model=model, datamodule=clap_datamodule, return_predictions=True)