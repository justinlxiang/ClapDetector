import torch

def mobilenet_v3(in_channels: int, num_classes: int):
    model = torch.hub.load('pytorch/vision:v0.11.0',
                           'mobilenet_v3_large', weights=None)

    model.features[0][0] = torch.nn.Conv2d(in_channels, 16, kernel_size=(
        3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    model.classifier[-1] = torch.nn.Linear(
        in_features=1280, out_features=num_classes)

    return model