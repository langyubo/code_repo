import cv2
import numpy as np

import torch
import torch.nn as nn
import timm
from functools import partial

device = "cuda:0" if torch.cuda.is_available() else "cpu"



class pretrained_CNN_ViT(nn.Module):
    def __init__(self, base, num_classes, num_patches, embed_dim, height=224, width=224, distilled=False, norm_layer=None, 
                 act_layer=None, drop_rate=0.):
        super(pretrained_CNN_ViT, self).__init__()
        # 参数初始化
        self.height = height
        self.width = width
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        
        # 模型定义
        self.cnn = base
        vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        self.vit_model = nn.Sequential(*list(vit_model.children())[2:])  #丢弃PatchEmbed层，直接输入(Bs, sequence_length, 768)

        

    def img_list_to_vec(self, img_list):
        '''
        input.shape = [bs, 5, 3, h, w]  5帧连续图像，rgb3通道， h,w为尺寸
        output.shape = [bs, sequence_length, 768]
        '''
        vec =[]   #tensor, [bs, sequence_length, 768]

        for i in range(img_list.shape[0]):
            imgs = img_list[i,:]  # tensor: [5, 3, h, w]

            v = self.cnn(imgs)  #tensor:[5, 768]
            vec.append(v)
        
        vec = torch.stack(vec, dim=0)  #[bs, sequence_length, 768]    
        return vec
    
    def forward_features(self, x):
        x = self.img_list_to_vec(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x
        # x = self.blocks(x)
        # x = self.norm(x)
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]

    def forward(self, x):      
        # x.shape: (bs, sequence_length=5, 3, h, w)
        vec = self.forward_features(x)  #[bs, sequence_length, 768]
        out = self.vit_model(vec)
        return out
