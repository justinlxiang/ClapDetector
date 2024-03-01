import json
import os
import numpy as np
import cv2
from tqdm import tqdm
import copy
import random
import torch
import yaml


def load_frame_data(data_config, set_type):
    images = []
    labels = []
                
    data_root_dir = data_config['label_root_dir']
    participants = data_config['participants']
    sessions = data_config['sessions'][set_type]
    for participant_type, participant_list in participants.items():
        for participant in participant_list:
            for session in os.listdir(os.path.join(data_root_dir, participant)):
                session_folder = os.path.join(data_root_dir, participant, session)
                
                if not os.path.exists(session_folder):
                    continue
                
                print(f"Processing {session_folder}")
                with open(os.path.join(session_folder, 'labels.txt'), 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    label, image_path = line.strip().split()
                    label = int(label)
                    images.append(image_path)
                    labels.append(label)

    # Convert images paths to tensor
    images_data = images
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return images_data, labels_tensor