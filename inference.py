import glob
import os
import re
from argparse import ArgumentParser
import json
import pytorch_lightning as pl
import yaml
from clap_dataset import ClapDataset, ClapDataModule

if __name__ == '__main__':
    pilot_data_path = "/data2/saif/eating/data/pilot_study"
    
    data_dict = {"ground_truth": {"videos": [], "syncing_poses": []}}

    for root, dirs, files in os.walk(pilot_data_path):
        for file in files:
            if file == "config.json":
                config_path = os.path.join(root, file)
                with open(config_path, 'r') as config_file:
                    config_data = json.load(config_file)
                    ground_truth = config_data.get("ground_truth", {})
                    video_file = ground_truth.get("videos")
                    syncing_pose = ground_truth.get("syncing_poses")
                    if video_file and syncing_pose is not None:
                        data_dict["ground_truth"]["videos"].append(video_file)
                        data_dict["ground_truth"]["syncing_poses"].append(syncing_pose)
    
    video_files = data_dict["ground_truth"]["videos"]
    labels = data_dict["ground_truth"]["syncing_poses"]
    clap_dataset = ClapDataset(video_files, labels)
    clap_datamodule = ClapDataModule(dataset=clap_dataset, batch_size=32, num_workers=4, split_ratio=0.8)

    # model prediction
    trainer = pl.Trainer(accelerator='gpu', devices=args.gpus)

    preds = trainer.predict(
        model=model, datamodule=clap_datamodule, return_predictions=True)