import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def resnet_encoder(model_name: str, input_channels: int, num_classes: int):
    if model_name not in model_urls:
        raise ValueError(f"{model_name} is not a valid model name. Choose from {list(model_urls.keys())}.")

    model = torch.hub.load('pytorch/vision:v0.11.0', model_name, pretrained=True)

    # Replace the last fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model