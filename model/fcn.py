#pip install pytorchcv --quiet
import torch
import torch.nn as nn

from pytorchcv.model_provider import get_model

# model = get_model("seresnext50_32x4d", pretrained=True)
# model = get_model("xception", pretrained=True)
# model = get_model("inceptionv3", pretrained=True)
# model = get_model("inceptionresnetv2", pretrained=True)
# model = get_model("mobilenet_w1", pretrained=True)
# model = get_model("efficientnet_b1", pretrained=True)
# model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer

# model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1)) # xcep
# model[0].final_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1)) #effi, incep

class Head(torch.nn.Module):
  def __init__(self, in_f, out_f):
    super(Head, self).__init__()
    
    self.f = nn.Flatten()
    self.l = nn.Linear(in_f, 512)
    self.d = nn.Dropout(0.75)
    self.o = nn.Linear(512, out_f)
    self.b1 = nn.BatchNorm1d(in_f)
    self.b2 = nn.BatchNorm1d(512)
    self.r = nn.ReLU()

  def forward(self, x):
    # torch.Size([32, 2048, 1, 1])
    x = self.f(x)
    # torch.Size([32, 2048])
    x = self.d(x)
    # torch.Size([32, 2048])

    x = self.l(x)
    # torch.Size([32, 512])
    x = self.r(x)
    # torch.Size([32, 512])
    x = self.d(x)
    # torch.Size([32, 512])

    out = self.o(x)
    # torch.Size([32, 2])
    return out

class FCN(torch.nn.Module):
  def __init__(self, base, in_f, out_f):
    super(FCN, self).__init__()
    self.base = base
    self.h1 = Head(in_f, out_f)
  
  def forward(self, x):
    # x.shape = torch.Size([32, 3, 224, 224])
    x = self.base(x)
    # x.shape = torch.Size([32, 2048, 1, 1])
    return self.h1(x)
    # x.shape = torch.Size([32, 2])

# model = FCN(model, 1280, 2) # effi
# model = FCN(model, 1536, 2) # incep-res
# model = FCN(model, 2048, 2) # xcep
