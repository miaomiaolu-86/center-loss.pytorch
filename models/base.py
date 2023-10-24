import torch
from torch import nn

from device import device

class FaceModel(nn.Module):

    '''
    初始化每个类的中心点
    num_classes个类
    每个样本特征维度是feature_dim
    '''
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        '''
        定义类的中心点center
        将数据copy到GPU运行
        在(num_classes, feature_dim)均匀分布种取随机数减0.5，×2，作为center的值
        定义分类器，将全连接层输出的多维数据变换成1维
        '''
        if num_classes:
            self.register_buffer('centers', (torch.rand(num_classes, feature_dim).to(device) - 0.5) * 2)
            self.classifier = nn.Linear(self.feature_dim, num_classes)
