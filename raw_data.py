# coding=utf-8
import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect # 二分查找模块
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler

class WavenetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_file,
                 item_length,
                 target_length,
                 classes=256,
                 mode="train"):

        #           |----receptive_field----|
        #                                 |--output_length--|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                           | | | | | | | | | |

        self.dataset_file = dataset_file    # 目标文件的路径
        self._item_length = item_length # ？
        self.target_length = target_length  # 输出音频的长度，输入音频长度-感受野长度+1
        self.classes = classes  # 256 每个采样点的离散化类别
        self.mode = mode

        # 从目标文件中读取量化后的音频数据
        self.data = np.load(self.dataset_file, mmap_mode='r')
        print("one hot input")



    def __getitem__(self, idx):
        # 将数据进行切片后保存成许多item
        file_data, label = self.data['arr_%d'%idx]
        example = torch.from_numpy(file_data).type(torch.LongTensor)
        one_hot = torch.FloatTensor(self.classes, self._item_length).zero_()
        one_hot.scatter_(0, example.unsqueeze(0), 1.)
        target = example[-self.target_length:].unsqueeze(0)
        return one_hot, target, label

    def __len__(self):
        return len(data.files)


def get_train_validation_data_loader(batch_size, random_seed, validation_size=0.3, shuffle=True,
                                     item_length=11264, target_length=5126, classes=256, mode = "train"):
    train_dataset = WavenetDataset(dataset_file='dataset/train_clip.npz',
                                   item_length=6139 + 5126 - 1,
                                   target_length=5126,
                                   classes=256,
                                   mode="train"
                                   )
    valid_dataset = WavenetDataset(dataset_file='dataset/train_clip.npz',
                                   item_length=6139 + 5126 - 1,
                                   target_length=5126,
                                   classes=256,
                                   mode="train"
                                   )
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_size * num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1
    )
    valid_loader = torch.utils.data.DataLoader(
          valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1
    )
    return train_loader, valid_loader
if __name__=='__main__':
    data = WavenetDataset(dataset_file='dataset/train_clip.npz',
                          item_length=6139 + 5126 - 1,
                          target_length=5126,
                          classes=256,
                          mode="train"
                          )