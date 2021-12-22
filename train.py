import math
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from transformer.vit_model import vit_base_patch16_224_in21k as create_model
from transformer.utils import read_split_data, train_one_epoch, evaluate

from dataset import Deepfakes
from sklearn.model_selection import train_test_split
from model.xception import Xception
from model.cross_vit import CrossViT
from model.resnet_attention import ResNet_50
from model.resnext_lstm import ResNext_LSTM
from model.fcn import *
from model.fcn_lstm import *

from pytorchcv.model_provider import get_model

from CNN_ViT_model import pretrained_CNN_ViT

def main():
   if torch.cuda.is_available() is False:
      raise EnvironmentError("not find GPU device for training.")

   # 基本参数设置
   num_class = 2
   lr = 0.0001
   lrf = 0.1
   epochs = 10
   batch_size = 8
   num_workers = 8
   model_weight_save_path = f"./weights/df_lr{lr}_epoch{epochs}_bs{batch_size}.pth"

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   print("using {} device.".format(device))
   
   label = list(np.load(r'/media/diode/WorkSpace/DataSets/Face Forensic/data_code/train_label_lite_30_224.npy'))
   data = list(np.load(r'/media/diode/WorkSpace/DataSets/Face Forensic/data_code/train_data_lite_30_224.npy'))
   

   # 划分train和val数据集
   x_train, x_val, y_train, y_val = train_test_split(data, label, shuffle=False)


   # 构建数据加载迭代器
   train_dataset = Deepfakes(x_train, y_train)
   val_dataset = Deepfakes(x_val, y_val)

   train_loader = DataLoader(dataset=train_dataset,
                             batch_size=batch_size, 
                             pin_memory=True,
                             num_workers=num_workers,
                             shuffle=True)
   val_loader = DataLoader(dataset=val_dataset,
                           batch_size=batch_size,
                           pin_memory=True,
                           num_workers=num_workers,
                           shuffle=True)

   # 搭建神经网络
   # 训练模式开启drop out，验证不开启
   train_mode = True
   # if train_mode:
   #    net = Xception(num_classes=num_class)
   #    net.fc = nn.Sequential(
   #            nn.Dropout(0.5),  # 以0.5的概率断开
   #            nn.LeakyReLU(inplace=True),
   #            nn.Linear(net.fc.in_features, num_class))
   #    net.to(device)
   # else:
   #    net = Xception(num_classes=num_class)
   #    net.fc = nn.Linear(net.fc.in_features, num_class)
   #    net.to(device)
   model = get_model("xception", pretrained=False)
   model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
   model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1)) # xcep

   if train_mode:
      cnn_net = LRCN(model, 2048, 768)
      cnn_net.to(device)
   else:
      cnn_net = LRCN(model, 2048, 768)
      cnn_net.to(device)

   #冻结xception
   for param in cnn_net.parameters():
      param.requires_grad=False

   net = pretrained_CNN_ViT(cnn_net, 2, 5, 768)
   net.to(device)
   
   # AdamW优化器
   pg = [p for p in net.parameters() if p.requires_grad]
   optimizer = optim.AdamW(pg, lr=lr, weight_decay=5E-5)

   # 余弦退火衰减，参考论文：Scheduler https://arxiv.org/pdf/1812.01187.pdf
   lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
   scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

   # 定义数组存储每一个epoch对应的acc和loss，用来监视训练过程曲线图
   Loss_list_train = []
   Accuracy_list_train = []

   Loss_list_val = []
   Accuracy_list_val = []

   best_acc = 0.0
   for epoch in range(epochs):
      # 开始训练
      train_mode = True
      train_loss, train_acc = train_one_epoch(model=net,
                                             optimizer=optimizer,
                                             data_loader=train_loader,
                                             device=device,
                                             epoch=epoch)
      scheduler.step()

      # 开始验证
      train_mode = False
      val_loss, val_acc = evaluate(model=net,
                                    data_loader=val_loader,
                                    device=device,
                                    epoch=epoch)

      print('[epoch %d] train_accuracy: %.3f  val_accuracy: %.3f  train_loss: %.3f  val_loss: %.3f' %
            (epoch + 1, train_acc, val_acc, train_loss, val_loss))

      # 储存最好的权重模型
      if val_acc > best_acc:
         best_acc = val_acc
         torch.save(net.state_dict(), model_weight_save_path)

      # 统计结果
      Loss_list_train.append(train_loss)
      Accuracy_list_train.append(train_acc)
      Loss_list_val.append(val_loss)
      Accuracy_list_val.append(val_acc)

   print('Finished Training')
   
   print(f"train_loss = {Loss_list_train}")
   print(f"train_acc = {Accuracy_list_train}")
   print(f"val_loss = {Loss_list_val}")
   print(f"val_acc = {Accuracy_list_val}")

   # # 训练结果可视化
   # x1 = range(0, epochs)
   # x2 = range(0, epochs)
   # y1 = Accuracy_list_train
   # y2 = Loss_list_train
   # plt.subplot(2, 1, 1)
   # plt.plot(x1, y1, 'o-')
   # plt.title('Train accuracy vs. epoches')
   # plt.ylabel('Train accuracy')
   # plt.subplot(2, 1, 2)
   # plt.plot(x2, y2, '.-')
   # plt.xlabel('Train loss vs. epoches')
   # plt.ylabel('Train loss')
   # plt.show()
   # plt.savefig('accuracy_loss_train.jpg')

   # x3 = range(0, epochs)
   # x4 = range(0, epochs)
   # y3 = Accuracy_list_val
   # y4 = Loss_list_val
   # plt.subplot(2, 1, 1)
   # plt.plot(x3, y3, 'o-')
   # plt.title('Val accuracy vs. epoches')
   # plt.ylabel('Val accuracy')
   # plt.subplot(2, 1, 2)
   # plt.plot(x4, y4, '.-')
   # plt.xlabel('Val loss vs. epoches')
   # plt.ylabel('Val loss')
   # plt.show()
   # plt.savefig('accuracy_loss_val.jpg')

if __name__ == '__main__':
    main()
