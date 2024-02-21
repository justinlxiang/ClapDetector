import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import cv2
from sklearn.model_selection import train_test_split
import os
import json
from data_generator import read_data_from_cache

class ClapDataset(Dataset):
    def __init__(self, datapoint_dicts, data_cache):
        self.datapoint_dicts = datapoint_dicts
        self.data_cache = data_cache
        
    def __len__(self):
        return len(self.datapoint_dicts)
    
    def __getitem__(self, idx):
        echo_profiles = dict()
        (
            video,
            frame,
        ) = read_data_from_cache(self.datapoint_dicts[idx], self.data_cache)

        video = torch.tesnor(np.array(video), dtype=torch.float)
        label_tensor = torch.tensor(frame, dtype=torch.long)
        return video, label_tensor


        