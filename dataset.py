import os
import pickle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import os
import PIL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import albumentations as A

def get_aug(img_arr):
    # 需进行两个操作：pip install albumentations
    # pip install -U albumentations[imgaug]
    trans = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussNoise(),  # 将高斯噪声添加到输入图像
            A.GaussNoise(),  # 将高斯噪声应用于输入图像。
        ], p=0.2),  # 应用选定变换的概率
        A.OneOf([
            A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
            A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
            A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
    ])
    trans_img = trans(image=img_arr)
    as_tensor = transforms.ToTensor()
    img_tensor = as_tensor(trans_img['image'])
    return img_tensor

def processing_img(image):
    image = transforms.functional.resize(image, (224, 224))
    image = get_aug(np.array(image))
    # image = transforms.functional.to_tensor(image)
    image = image.numpy()[::-1].copy()
    image = torch.from_numpy(image)
    image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image

class Deepfakes(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        # 统计数据集分布情况
        self.get_static()

    def get_static(self):
        num_fake, num_real, wrong = 0, 0, 0
        for per_label in self.label:
            if per_label == '0':
                num_fake += 1
            elif per_label == "1":
                num_real += 1
            else:
                wrong += 1
        print(f"num of fake: {num_fake}, num of real: {num_real}, num of wrong: {wrong}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_list = self.data[index]
        imgs = []
        for img in img_list:
            image_ori = Image.fromarray(img)
            image = processing_img(image_ori)
            imgs.append(image)
        label = int(self.label[index])  # 把字符串标签转换为整数标签
        label = torch.from_numpy(np.array(label))  # 先把整数转np数组再转torch的tensor
        imgs = torch.stack(imgs)
        data = (imgs, label)    
        return data

if __name__ == '__main__':
    dataset = ICCV(list(np.load(r'./0train_data_50_c40.npy')), list(np.load(r'./0train_label_50_c40.npy')))
    print("a")

    # img_tiny_path = "tinyDataset/img_tiny_train/"
    # dataset = ICCV(img_tiny_path)
    # train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True,
    #                           num_workers=0, pin_memory=False)
    #
    # for i, data in enumerate(train_loader):
    #     img = data[0]
    #     label = data[1]
    #     print("end")
