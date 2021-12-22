# !pip install pytorchcv --quiet

from sys import int_info
import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model

model = get_model("xception", pretrained=True)
# model = get_model("efficientnet_b1", pretrained=True)
model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
# model[0].final_pool = nn.AdaptiveAvgPool2d(1)
model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))

class Head(torch.nn.Module):
    def __init__(self, in_f):
      super(Head, self).__init__()
      
      self.f = nn.Flatten()
      self.l = nn.Linear(in_f, 256)
      self.d = nn.Dropout(0.75)
      self.b1 = nn.BatchNorm1d(in_f)
      self.b2 = nn.BatchNorm1d(256)
      self.r = nn.ReLU()

    def forward(self, x):
      # torch.Size([32, 2048, 1, 1])
      x = self.f(x)
      # torch.Size([32, 2048])
      x = self.d(x)
      # torch.Size([32, 2048])

      x = self.l(x)
      # torch.Size([32, 256])
      x = self.r(x)
      # torch.Size([32, 256])
      x = self.d(x)
      # torch.Size([32, 256])

      return x

class LRCN(nn.Module):
    def __init__(self, base, in_f, out_f):
        super(LRCN, self).__init__()
        self.cnn = base
        self.head = Head(in_f)
        
        self.LSTM = nn.LSTM(
            input_size=in_f,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.f1 = nn.Linear(256, 128)
        self.f2 = nn.Linear(128, out_f)
        self.r = nn.ReLU()
        self.d = nn.Dropout(0.5)
        
    def forward(self, x):
        # what we get --> x.shape = torch.Size([32, 3, 224, 224])
        timesteps = 1
        batch_size, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        # what we get --> x.shape = torch.Size([32, 3, 224, 224])
        x = self.cnn(x)
        x2 = self.head(x)
        # what we get --> x.shape = torch.Size([32, 2048, 1, 1])
        x = x.view(batch_size, timesteps, -1)
        # what we get --> x.shape = torch.Size([32, 1, 2048])
        self.LSTM.flatten_parameters()
        x, (hn,hc) = self.LSTM(x)
        # what we get --> x.shape = torch.Size([32, 1, 256])
        x = self.d(self.r(x[:,-1,:]))
        # torch.Size([32, 256])

        x_total = x + x2
        x_total = self.f1(x_total)
        x_total = self.f2(x_total)
        return x_total
        # torch.Size([32, 2])

# model = LRCN(model, 2048, 2)
