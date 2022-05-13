from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        '''
        Args:
            x:一个 shape 为 [batch_size, c, h, w] 的浮点型 tensor

        Returns:一个 shape 为 [batch_size, c*h*w] 的浮点型 tensor
        '''
        # 这里要将 (h, w) 转成 (w, h)，这是因为我们返回的点要是 (x, y)，便于后续操作
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class PNet(nn.Module):
    def __init__(self):

        super(PNet, self).__init__()

        # 输入：(H, W)
        # 第一层卷积结果: H - 2,
        # 池化结果: ceil((H - 2)/2),
        # 第二层卷积结果: ceil((H - 2)/2) - 2,
        # 第三层卷积结果: ceil((H - 2)/2) - 4,
        # W 的变化一样

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 10, 3, 1)),
                    ("prelu1", nn.PReLU(10)),
                    ("pool1", nn.MaxPool2d(2, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(10, 16, 3, 1)),
                    ("prelu2", nn.PReLU(16)),
                    ("conv3", nn.Conv2d(16, 32, 3, 1)),
                    ("prelu3", nn.PReLU(32)),
                ]
            )
        )

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        dir_path = path.dirname(__file__)
        #将 <class 'numpy.ndarray'> 转成 <class 'dict'>
        weights = np.load(path.join(dir_path, "weights/pnet.npy"), allow_pickle=True)[()]
        #网络层的名字和参数的迭代器
        for layer, params in self.named_parameters():
            params.data = torch.FloatTensor(weights[layer])

    def forward(self, x):
        """
        Args:
            x:一个 shape 为 [batch_size, 3, h, w] 的浮点型 tensor

        Returns:
            box:一个 shape 为 [batch_size, 4, h', w'] 的浮点型 tensor
            cls:一个 shape 为 [batch_size, 2, h', w'] 的浮点型 tensor
        """
        x = self.features(x)
        cls = self.conv4_1(x)
        box = self.conv4_2(x)
        cls = F.softmax(cls, dim=1)
        return box, cls


class RNet(nn.Module):
    def __init__(self):

        super(RNet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 28, 3, 1)),
                    ("prelu1", nn.PReLU(28)),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(28, 48, 3, 1)),
                    ("prelu2", nn.PReLU(48)),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv3", nn.Conv2d(48, 64, 2, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("flatten", Flatten()),
                    ("conv4", nn.Linear(576, 128)),
                    ("prelu4", nn.PReLU(128)),
                ]
            )
        )

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        dir_path = path.dirname(__file__)
        weights = np.load(path.join(dir_path, "weights/rnet.npy"), allow_pickle=True)[()]
        for layer, params in self.named_parameters():
            params.data = torch.FloatTensor(weights[layer])

    def forward(self, x):
        """
            Args:
                x:一个 shape 为 [batch_size, 3, h, w] 的浮点型 tensor

            Returns:
                box:一个 shape 为 [batch_size, 4] 的浮点型 tensor
                cls:一个 shape 为 [batch_size, 2] 的浮点型 tensor
        """
        x = self.features(x)
        cls = self.conv5_1(x)
        box = self.conv5_2(x)
        cls = F.softmax(cls, dim=1)
        return box, cls

class ONet(nn.Module):
    def __init__(self):

        super(ONet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, 3, 1)),
                    ("prelu1", nn.PReLU(32)),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(32, 64, 3, 1)),
                    ("prelu2", nn.PReLU(64)),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv3", nn.Conv2d(64, 64, 3, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("pool3", nn.MaxPool2d(2, 2, ceil_mode=True)),
                    ("conv4", nn.Conv2d(64, 128, 2, 1)),
                    ("prelu4", nn.PReLU(128)),
                    ("flatten", Flatten()),
                    ("conv5", nn.Linear(1152, 256)),
                    ("drop5", nn.Dropout(0.25)),
                    ("prelu5", nn.PReLU(256)),
                ]
            )
        )

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        dir_path = path.dirname(__file__)
        weights = np.load(path.join(dir_path, "weights/onet.npy"), allow_pickle=True)[()]
        for layer, params in self.named_parameters():
            params.data = torch.FloatTensor(weights[layer])

    def forward(self, x):
        """
            Args:
                x:一个 shape 为 [batch_size, 3, h, w] 的浮点型 tensor

            Returns:
                box:一个 shape 为 [batch_size, 4] 的浮点型 tensor
                cls:一个 shape 为 [batch_size, 2] 的浮点型 tensor
                landmark:一个 shape 为 [batch_size, 10] 的浮点型 tensor
        """
        x = self.features(x)
        cls = self.conv6_1(x)
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)
        cls = F.softmax(cls, dim=1)
        return landmark, box, cls