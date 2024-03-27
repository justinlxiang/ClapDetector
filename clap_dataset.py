import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import cv2
from sklearn.model_selection import train_test_split
import os
import json
from data_generator import load_frame_data
import yaml

class ClapDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        image_path = self.frames[idx]
        label = self.labels[idx]

        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (224, 224))
        frame = np.array(frame, dtype=np.uint8)

        frame = torch.tensor(frame, dtype = torch.float)
        # label_tensor = label

        return frame, label
    

class ClapDataModule(pl.LightningDataModule):
    def __init__(self, data_config, batch_size=32, sliding_window_size=3, num_workers=4, split_ratio=0.8):
        super().__init__()
        self.data_config= data_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.sliding_window_size = sliding_window_size

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.label_dict = None

    def get_label_dict(self):
        if self.label_dict is not None:
            return self.label_dict
        else:
            return "Label dictionary NOT found. Call the setup() function of the datamodule first"

    def setup(self, stage=None):
        # Load dataset into train, validation, and test sets
        train_frames, train_labels = load_frame_data(self.data_config, 'train')
        val_frames, val_labels = load_frame_data(self.data_config, 'val')
        test_frames, test_labels = load_frame_data(self.data_config, 'test')

        if self.data_config["random_split"]:
            all_frames = train_frames + val_frames + test_frames
            all_labels = train_labels + val_labels + test_labels

            total_samples = len(all_frames)
            train_size = int(total_samples * self.split_ratio)
            val_size = int((total_samples - train_size) / 2)
            test_size = total_samples - train_size - val_size

            train_frames, temp_frames, train_labels, temp_labels = train_test_split(all_frames, all_labels, train_size=train_size, shuffle=True)
            val_frames, test_frames, val_labels, test_labels = train_test_split(temp_frames, temp_labels, test_size=test_size, shuffle=True)

        self.train_set = ClapDataset(train_frames, train_labels)
        self.val_set = ClapDataset(val_frames, val_labels)
        self.test_set = ClapDataset(test_frames, test_labels)
        
        # self.label_dict = {frame: label() for frame, label in zip(train_frames, train_labels)}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    
    data_config_file = open("configs/data.yaml", mode="r")
    data_cfg = yaml.load(data_config_file, Loader=yaml.FullLoader)

    clap_data_module = ClapDataModule(data_config=data_cfg, batch_size=8, num_workers=4, split_ratio=0.8)


