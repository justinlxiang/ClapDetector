import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

# from models.classification.cnn_estimator import ResNetClassifier
# from models.resnet.branched_encoder import BranchedResNet
from models.resnet.encoder import resnet_encoder
from models.mobilenet.encoder import mobilenet_v3
from models.losses.focal_loss import FocalLoss
# from utils.lightning_hooks import BackwardHook
# from torchray.attribution.grad_cam import grad_cam
# from datareader.utils import plot_profiles, label_dict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import uuid

class ConvModel(pl.LightningModule):
    def __init__(
        self,
        model_architecture: str,
        input_channels: int,
        output_dim: int,
        learning_rate: float,
        loss_function: str,
        class_weights: list,
        lr_scheduler: str,
        **kwargs,
    ):
        super().__init__()

        self.kwargs = kwargs
        
        self.model_architecture = model_architecture
        self.loss_function = loss_function
        self.class_weights = class_weights

        self.input_channels = input_channels
        self.output_dim = output_dim

        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler

        if self.model_architecture == "resnet18" or self.model_architecture == "resnet34" or "resnet" in self.model_architecture:
            self.cnn_model = resnet_encoder(model_name=self.model_architecture, input_channels=self.input_channels, num_classes=self.output_dim)

        elif self.model_architecture == "mobilenet_v3":
            self.cnn_model = mobilenet_v3(in_channels=self.input_channels, num_classes=self.output_dim)
        
         # loss function definition
        if self.loss_function == "cross-entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_function == "focal-loss":
            alpha = np.array(self.output_dim * [1.0])

            if len(self.class_weights) > 0:
                self.class_weights = list(
                    filter(lambda x: x < (self.output_dim - 1), self.class_weights)
                )
                alpha[self.class_weights] = 2.5

            alpha = alpha / np.sum(alpha)
            self.criterion = FocalLoss(alpha=alpha.tolist(), gamma=2)

        # metric
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=self.output_dim, average="macro"
        )
        self.micro_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=self.output_dim, average="micro"
        )
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.output_dim)
        
        self.save_hyperparameters()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.cnn_model(x)
        x = torch.softmax(x, dim=1)

        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler == "multistep_lr":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[35, 40, 45], gamma=0.1
            )
        elif self.lr_scheduler == "cosine_annealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
            )

        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        loss = self.criterion(y_pred, y)
        f1_score = self.f1_score(y_pred, y)
        micro_f1 = self.micro_f1(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log("train_loss", loss)
        self.log("train_f1", f1_score, on_step=False, on_epoch=True)
        self.log("train_microf1", micro_f1, on_step=False, on_epoch=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        val_loss = self.criterion(y_pred, y)
        f1_score = self.f1_score(y_pred, y)
        micro_f1 = self.micro_f1(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log("val_loss", val_loss)
        self.log("val_macrof1", f1_score, on_step=False, on_epoch=True)
        self.log("val_microf1", micro_f1, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)
        

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        test_loss = self.criterion(y_pred, y)
        f1_score = self.f1_score(y_pred, y)
        micro_f1 = self.micro_f1(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log("test_loss", test_loss)
        self.log("test_f1", f1_score)
        self.log("test_microf1", micro_f1, on_step=False, on_epoch=True)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_pred = self(x)

        return {"pred": torch.argmax(y_pred, dim=1), "truth": y}