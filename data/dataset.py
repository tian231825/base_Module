# -*- encoding: utf-8 -*-
"""
@File    : dataset.py
@Time    : 2022/4/22 11:38
@Author  : junruitian
@Software: PyCharm
"""

import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import transforms


class DogCat(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):

        '''
        根据目标图片路径 并根据训练 验证 测试划分数据

        :param root:
        :param transform:
        :param train:
        :param test:
        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg

        '''
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split(".")[-2].split("/")[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split(".")[-2]))

        imgs_num = len(imgs)

        # shuffle imgs
        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        # 划分训练、验证集, 验证：训练=3：7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.7*imgs_num):]

        if transforms is None:
            # 数据转换操作 测试集 验证集 和训练集的转换有所区别
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if self.test or not train:
                self.transforms = T.Compose([T.Scale(224), T.CenterCrop(224), T.ToTensor(), normalize])
            else:
                self.transforms = T.Compose([T.Scale(256), T.RandomSizedCrop(224), T.RandomHorizontalFlip(),
                                             T.ToTensor(), normalize])

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        对于测试集 没有label 返回图片id
        :param index:
        :return:
        '''
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split(".")[-2].split("/")[-1])
        else:
            label = 1 if 'dog' in img_path.split("/")[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        '''

        返回数据集中图片数量
        :return:
        '''
        return len(self.imgs)