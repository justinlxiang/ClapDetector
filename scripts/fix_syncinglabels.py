participants = ['R02_hand_02', 'R02_wild', 'R02_wild_03', 'R04-hand2']
import os
import json

base_path = '/data2/saif/eating/data/pilot_study'
for participant in participants:
    config_path = os.path.join(base_path, participant, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config_data = json.load(file)
        
        if 'ground_truth' in config_data and 'syncing_poses' in config_data['ground_truth']:
            config_data['ground_truth']['syncing_poses'] = [pose + 1 for pose in config_data['ground_truth']['syncing_poses']]
            
            with open(config_path, 'w') as file:
                json.dump(config_data, file, indent=4)
