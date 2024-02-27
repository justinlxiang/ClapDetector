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


def create_frame_session_folders(data_config):
    data_root_dir = data_config['data_root_dir']
    participants = data_config['participants']

    labels_root = "./labels_folder"
    frames_root = "./frame_folder"
    if not os.path.exists(labels_root):
        os.makedirs(labels_root)
    if not os.path.exists(frames_root):
        os.makedirs(frames_root)

    for participant_type, participant_list in participants.items():
        for participant in participant_list:
            participant_path = os.path.join(data_root_dir, participant_type, participant)
            config_path = os.path.join(participant_path, "config.json")
            print(participant)
            if not os.path.exists(os.path.join(labels_root, participant)):
                os.mkdir(os.path.join(labels_root, participant))
                os.mkdir(os.path.join(frames_root, participant))

            participant_labels_folder = os.path.join(labels_root, participant)
            participant_frames_folder = os.path.join(frames_root, participant)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    ground_truths = config.get("ground_truth", [])
                    videos = ground_truths.get("videos", [])
                    syncing_poses = ground_truths.get("syncing_poses", [])
                    print(videos, syncing_poses)
                    for i, syncing_pose in enumerate(syncing_poses):
                        if not os.path.exists(os.path.join(participant_labels_folder, f"session_{i+1:02d}")):
                            print("Processing Participant " + participant + " Session " + str(i+1))
                            video = videos[i]
                            
                            os.mkdir(os.path.join(participant_labels_folder, f"session_{i+1:02d}"))
                            labels_folder = os.path.join(participant_labels_folder, f"session_{i+1:02d}")
                            
                            os.mkdir(os.path.join(participant_frames_folder, f"session_{i+1:02d}"))
                            frames_folder = os.path.join(participant_frames_folder, f"session_{i+1:02d}")
        
                            video_path = os.path.join(participant_path, video) 
                            cap = cv2.VideoCapture(video_path)
                            frame_count = 1
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    print("No frame at frame: " + str(frame_count))
                                    break
                                frame = cv2.resize(frame, (224, 224))
                                frame_path = os.path.join(frames_folder, f"participant_{participant}_session_{i+1:02d}_frame_{frame_count:04d}.jpg")
                                cv2.imwrite(frame_path, frame)
                                frame_count += 1
                            cap.release()
                            
                            labels_path = os.path.join(labels_folder, "labels.txt")
                            with open(labels_path, 'w') as labels_file:
                                frame_files = os.listdir(frames_folder)
                                for frame_index, frame_file in enumerate(frame_files):
                                    frame_file_path = os.path.abspath(os.path.join(frames_folder, frame_file))
                                    if participant_type == "pilot_frame_off":
                                        label = 1 if frame_index+1 == syncing_pose+1 else 0
                                    else:
                                        label = 1 if frame_index+1 == syncing_pose else 0
                                    labels_file.write(f"{label} {frame_file_path}\n")


if __name__ == "__main__":
    data_config_file = open("configs/data.yaml", mode="r")
    data_cfg = yaml.load(data_config_file, Loader=yaml.FullLoader)
    create_frame_session_folders(data_cfg)
