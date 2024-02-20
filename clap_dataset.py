import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import cv2
from sklearn.model_selection import train_test_split
import os
import json

class ClapDataset(Dataset):
    def __init__(self, video_files, labels):
        self.video_files = video_files
        self.labels = labels
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video = np.zeros((frame_count, 224, 224, 3), dtype = np.uint8)
        for i in range(frame_count):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (224, 224))
            video[i] = frame
        cap.release()

        # Convert frames to numpy array
        video_data = torch.tensor(video, dtype = torch.float)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return video_data, label_tensor
    

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

    def setup(self, stage=None):
        # Split dataset into train, validation, and test
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        train_indices, test_indices = train_test_split(indices, test_size=1-self.split_ratio, random_state=42)
        val_size = 0.5 * (1 - self.split_ratio)
        train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=42)

        train_video_files = [self.dataset.video_files[i] for i in train_indices]
        train_labels = [self.dataset.labels[i] for i in train_indices]
        self.train_set = ClapDataset(train_video_files, train_labels)

        val_video_files = [self.dataset.video_files[i] for i in val_indices]
        val_labels = [self.dataset.labels[i] for i in val_indices]
        self.val_set = ClapDataset(val_video_files, val_labels)

        test_video_files = [self.dataset.video_files[i] for i in test_indices]
        test_labels = [self.dataset.labels[i] for i in test_indices]
        self.test_set = ClapDataset(test_video_files, test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":

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
    
    # placeholder example input
    # data_dict = {
    #     "ground_truth": {
    #         "videos": [
    #             "record_20230605_121225_158207.mp4",
    #             "record_20230605_130703_324246.mp4",
    #             "record_20230606_125009_189312.mp4",
    #             "record_20230606_131806_228554.mp4"
    #         ],
    #         "syncing_poses": [
    #             213,
    #             98,
    #             140,
    #             91
    #         ]
    #     }
    # }
    
    video_files = data_dict["ground_truth"]["videos"]
    labels = data_dict["ground_truth"]["syncing_poses"]
    clap_dataset = ClapDataset(video_files, labels)
    clap_data_module = ClapDataModule(dataset=clap_dataset, batch_size=32, num_workers=4, split_ratio=0.8)

