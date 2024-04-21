import torch

def mobilenet_v3(in_channels: int, num_classes: int):
    model = torch.hub.load('pytorch/vision:v0.11.0',
                           'mobilenet_v3_large', weights = True)

    model.classifier[-1] = torch.nn.Linear(
        in_features=1280, out_features=num_classes)

    return model