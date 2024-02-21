import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import cv2
from sklearn.model_selection import train_test_split
import os
import json
from clap_dataset import ClapDataset
from data_generator import generate_data

class ClapDataModule(pl.LightningDataModule):
    def __init__(self, data_config: dict, sliding_window_size : int, batch_size=32, num_workers=4, split_ratio=0.8):
        super().__init__()
        self.data_config = data_config
        self.sliding_window_size = sliding_window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.label_dict = None
        self._already_called = False

    def get_label_dict(self):
            if self.label_dict is not None:
                return self.label_dict
            else:
                return "Label dictionary NOT found. Call the setup() function of the datamodule first"
            
    def get_random_split(self, all_dicts):
        all_dicts = train_dicts + val_dicts
        train_dicts, val_dicts = train_test_split(
            all_dicts,
            test_size=0.2,
            random_state=3407,
            stratify=[d["label"] for d in all_dicts],
        )

        return (train_dicts, val_dicts)

    def setup(self, stage=None):
        if self._already_called:
            return
        
        (all_data_dicts, all_data_caches, all_label_dicts) = generate_data(
            data_config=self.data_config,
            sliding_window_size=self.sliding_window_size
        )

        if self.data_config["random_split"]:
            train_data_dicts, val_data_dicts = self.get_random_split(all_data_dicts)
            files_in_train = [d["video_folder_path"] for d in train_data_dicts]
            files_in_val = [d["video_folder_path"] for d in val_data_dicts]

            # Take subset of caches corresponding to keys in data_dicts
            train_data_cache = {k: v for k, v in all_data_caches.items() if k in files_in_train}
            val_data_cache = {k: v for k, v in all_data_caches.items() if k in files_in_val}

            train_labels = [d["label"] for d in train_data_dicts]
            val_labels = [d["label"] for d in val_data_dicts]
            train_label_dict = {k: v for k, v in all_label_dicts.items() if k in train_labels}
            val_label_dict = {k: v for k, v in all_label_dicts.items() if k in val_labels}
        
        test_data_dicts = val_data_dicts
        test_data_cache = val_data_cache
        
        # Split dataset into train, validation, and test
        self.train_set = ClapDataset(train_data_dicts, train_data_cache)
        self.val_set = ClapDataset(val_data_dicts, val_data_cache)
        self.test_set = ClapDataset(test_data_dicts, test_data_cache)


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)


