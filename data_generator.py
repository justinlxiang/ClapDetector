import json
import os
import numpy as np
import cv2
import tqdm
import copy
import random
import torch
import yaml


def load_frame_data(data_config, set_type):
    images = []
    labels = []
                
    data_root_dir = data_config['data_root_dir']
    participants = data_config['participants']
    sessions = data_config['sessions'][set_type]

    for participant_type, participant_list in participants.items():
        for participant in participant_list:
            for session in sessions:
                session_folder = os.path.join(data_root_dir, participant_type, participant, session)
                
                if not os.path.exists(session_folder):
                    continue
                
                print(f"Processing {session_folder}")
                with open(os.path.join(session_folder, 'labels.txt'), 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    label, image_path = line.strip().split()
                    label = int(label)
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (224, 224))
                    image = np.array(image, dtype=np.uint8)
                    images.append(image)
                    labels.append(label)

    # Convert images to tensor
    images_data = torch.tensor(images, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return images_data, labels_tensor
