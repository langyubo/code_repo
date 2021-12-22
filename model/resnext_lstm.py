from re import X
import torch
from torch import nn
from torchvision import models

class ResNext_LSTM(nn.Module):
    def __init__(self, num_class, image_size=224, latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(ResNext_LSTM, self).__init__()
        self.image_size = image_size
        model = models.resnext50_32x4d(pretrained = False)  # Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_class)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size = x.shape[0]
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, -1, 2048)
        x_lstm,_ = self.lstm(x, None)
        return self.dp(self.linear1(torch.mean(x_lstm, dim = 1)))