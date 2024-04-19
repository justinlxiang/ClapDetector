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

    clap_count = 0
    non_clap_count = 0
                
    data_root_dir = data_config['label_root_dir']
    participants = data_config['participants']
    split_data = data_config['split'][set_type]
    for participant in split_data:
        for session in os.listdir(os.path.join(data_root_dir, participant)):
            session_folder = os.path.join(data_root_dir, participant, session)
            
            if not os.path.exists(session_folder):
                continue
            
            # print(f"Processing {session_folder}")
            with open(os.path.join(session_folder, 'labels.txt'), 'r') as f:
                lines = f.readlines()
            
            additional_clap_frames = 10
            sampling_rate = 10 # how many frames per 1 frame saved

            for i in range(len(lines)):
                line = lines[i]
                label, image_path = line.strip().split()
                # print(i, label)
                label = int(label)
                if label == 1 or i % sampling_rate == 0:
                    if label == 1:
                        for j in range(i, i + additional_clap_frames):
                            line = lines[j]
                            label, image_path = line.strip().split()
                            images.append(image_path)
                            labels.append(1)
                            clap_count += 1
                        break
                    images.append(image_path)
                    labels.append(label)
                    non_clap_count += 1
    
    print(set_type + "Clap Frames: " + str(clap_count))
    print(set_type + "Non Clap Frames: " + str(non_clap_count))


    # Convert images paths to tensor
    images_data = images
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return images, labels