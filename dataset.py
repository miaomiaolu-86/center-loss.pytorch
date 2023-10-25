import os
import random
import tarfile
from math import ceil, floor

from torch.utils import data
import numpy as np

from utils import image_loader, download

DATASET_TARBALL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
PAIRS_TRAIN = "http://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt"
PAIRS_VAL = "http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt"

#构建训练和验证数据集
def create_datasets(dataroot, train_val_split=0.9):
    #判断给定路径是否为目录
    if not os.path.isdir(dataroot):
        os.mkdir(dataroot)

    #列出目录下所有文件
    dataroot_files = os.listdir(dataroot)
    #获取路径最后一个/后的字符串：lfw-deepfunneled.tgz
    data_tarball_file = DATASET_TARBALL.split('/')[-1]
    #获取data_tarball_file第一个.之前的字符串：lfw-deepfunneled
    #获取数据集文件名称
    data_dir_name = data_tarball_file.split('.')[0]

    '''
    判断数据集文件是否在给定路径的文件目录里
    不在的话，重新下载并解压文件到目录
    '''
    if data_dir_name not in dataroot_files:
        if data_tarball_file not in dataroot_files:
            tarball = download(dataroot, DATASET_TARBALL)
        with tarfile.open(tarball, 'r') as t:
            t.extractall(dataroot)

    #拼接路径，获取文件目录
    images_root = os.path.join(dataroot, 'lfw-deepfunneled')
    #列出该文件下所有文件/文件夹
    names = os.listdir(images_root)
    #判断目录下文件是否为空
    if len(names) == 0:
        raise RuntimeError('Empty dataset')

    #初始化空的训练和验证数据集
    training_set = []
    validation_set = []
    #enumerate()将数据组合为一个索引序列，同时列出数据和数据下标
    for klass, name in enumerate(names):
        def add_class(image):
            #拼接含有数据和数据下标的图像路径
            image_path = os.path.join(images_root, name, image)
            return (image_path, klass, name)

        #列出所有具体名字的图像
        images_of_person = os.listdir(os.path.join(images_root, name))
        #图像大小
        total = len(images_of_person)

        '''图像大小*0.9，向上取整作为训练数据集
        图像大小*0.9，向下取整作为验证数据集'''
        training_set += map(
                add_class,
                images_of_person[:ceil(total * train_val_split)])
        validation_set += map(
                add_class,
                images_of_person[floor(total * train_val_split):])

    return training_set, validation_set, len(names)


class Dataset(data.Dataset):

    def __init__(self, datasets, transform=None, target_transform=None):
        self.datasets = datasets
        self.num_classes = len(datasets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        image = image_loader(self.datasets[index][0])
        if self.transform:
            image = self.transform(image)
        return (image, self.datasets[index][1], self.datasets[index][2])


class PairedDataset(data.Dataset):

    def __init__(self, dataroot, pairs_cfg, transform=None, loader=None):
        self.dataroot = dataroot
        self.pairs_cfg = pairs_cfg
        self.transform = transform
        self.loader = loader if loader else image_loader

        self.image_names_a = []
        self.image_names_b = []
        self.matches = []

        self._prepare_dataset()

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, index):
        return (self.transform(self.loader(self.image_names_a[index])),
                self.transform(self.loader(self.image_names_b[index])),
                self.matches[index])

    def _prepare_dataset(self):
        raise NotImplementedError


class LFWPairedDataset(PairedDataset):

    def _prepare_dataset(self):
        pairs = self._read_pairs(self.pairs_cfg)

        for pair in pairs:
            if len(pair) == 3:
                match = True
                name1, name2, index1, index2 = \
                    pair[0], pair[0], int(pair[1]), int(pair[2])

            else:
                match = False
                name1, name2, index1, index2 = \
                    pair[0], pair[2], int(pair[1]), int(pair[3])

            self.image_names_a.append(os.path.join(
                    self.dataroot, 'lfw-deepfunneled',
                    name1, "{}_{:04d}.jpg".format(name1, index1)))

            self.image_names_b.append(os.path.join(
                    self.dataroot, 'lfw-deepfunneled',
                    name2, "{}_{:04d}.jpg".format(name2, index2)))
            self.matches.append(match)

    def _read_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return pairs
