import json
import os
import numpy as np
import cv2
import tqdm
import copy
import random

def generate_data(data_config, sliding_window_size):
    data_dicts = []
    label_dict = dict()
    num_classes = data_config["num_classes"]

def read_data_from_cache(data_dict, data_cache):
    return