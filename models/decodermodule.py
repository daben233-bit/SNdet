import torch
from torch import nn
import torch.nn.init as init
import math
import numpy as np
# from models import *
import torch.nn.functional as F

class _GAblock(nn.Module):
    def __init__(self, channel_low, channel_high):
        '''
        Global Attention 模块
        :param channel_low: feature_low 的通道数
        :param channel_high: feature_high 的通道数
        '''
        super(_GAblock, self).__init__()
        self.f_low = nn.Sequential(
            nn.Conv2d(channel_low, channel_high, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(channel_high),
            nn.ReLU(inplace=True)
        )
        self.f_high = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(channel_high, channel_high, kernel_size=1)
        )
        # self.upsample = nn.Upsample(size=size_low, mode='bilinear', align_corners=True)
        self.upsample = nn.ConvTranspose2d(channel_high, channel_high, kernel_size=4, stride=2, padding=1)
        self._initialize_weights()

    def forward(self, feature_low, feature_high):
        # size = feature_low.size()[2:]
        feature_low_conv = self.f_low(feature_low)
        #feature_up = self.upsample(feature_high)
        feature_high_GP = self.f_high(feature_high)
        # feature_high_upsample = F.upsample(feature_high, size=size, mode='bilinear', align_corners=True)
        feature_high_upsample = self.upsample(feature_high)
        # print(feature_high_upsample.size())
        # feature_high 1, 512, 1, 1        feature_low and upsample  1, 512, 10, 10
        feature_fused = feature_high_GP * feature_low_conv + feature_high_upsample
        # print(feature_fused.size())
        return feature_fused

        # feature_high 1, 512, 1, 1        feature_low and upsample  1, 512, 10, 10
    def _initialize_weights(self):
       for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

class _SPPblock(nn.Module):
    '''
    Spatial Pyramid Pooling模块
    '''
    def __init__(self, sizes=None):
        super(_SPPblock, self).__init__()
        if sizes is None:
            sizes = [1, 3, 6, 8]
        self.stages = nn.ModuleList(nn.AdaptiveAvgPool2d(output_size=(size, size)) for size in sizes)

    def forward(self, x):
        n, c, _, _ = x.size()
        priors = [stage(x).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center
    

class _NLblock(nn.Module):
    '''
    Non-Local Attention 模块
    '''
    def __init__(self, channel_ga):
        super(_NLblock, self).__init__()
        self.f_phi = nn.Sequential(
            nn.Conv2d(channel_ga, 256, kernel_size=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.f_theta = nn.Sequential(
            nn.Conv2d(channel_ga, 256, kernel_size=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.f_gamma = nn.Conv2d(channel_ga, 256, kernel_size=1)
        self.sample1 = _SPPblock()
        self.sample2 = _SPPblock()
        self.W = nn.Conv2d(256, channel_ga, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):
        batch_size, channel, h, w = x.size()  # C = channel
        phi = self.f_phi(x)  # batch_size C W H     C = 256
        theta = self.f_theta(x)  # batch_size C W H
        gamma = self.f_gamma(x)  # batch_size C W H
        theta_p = self.sample1(theta)  # batch_size C S
        gamma_p = self.sample2(gamma)  # batch_size C S
        phi_reshape = torch.reshape(phi, (batch_size, h * w, 256))  # batch_size W*H C
        y1 = torch.matmul(phi_reshape, theta_p)  # (batch_size W*H C) * (batch_size C S) = (batch_size W*H S)
        # print(y1.size())
        y1_sq = ((h * w) ** -.25) * y1
        y1_norm = F.softmax(y1_sq, dim=-1)  # normalized

        gamma_p = gamma_p.permute(0, 2, 1)  # batch_size C S ---> batch_size S C
        y2 = torch.matmul(y1_norm, gamma_p)  # (batch_size W*H S) * (batch_size S C) = (batch_size W*H C)
        y2_reshape = torch.reshape(y2, (batch_size, 256, h, w))  # batch_size C W H  C = 256
        y2_w = self.W(y2_reshape)
        y = x + y2_w
        return y
    
    def _initialize_weights(self):
        for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal(m.weight)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant(m.weight, 1)
                    nn.init.constant(m.bias, 0)

class _SAblock(nn.Module):
    '''
    Spatial Attention 模块
    '''
    def __init__(self, channel_nl):
        super(_SAblock, self).__init__()
        self.conv1 = nn.Conv2d(channel_nl, channel_nl, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_nl, channel_nl, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x):
        conv1 = self.relu1(self.conv1(x))
        sattmap = torch.sigmoid(self.conv2(conv1))
        O_s = sattmap * conv1
        y = x + O_s
        return y

    def _initialize_weights(self):
       for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

class CAModule(nn.Module):
    '''
    Context Attention 模块
    '''
    def __init__(self, channel_low, channel_high):
        super(CAModule, self).__init__()
        self.channel_low = channel_low
        self.channel_high = channel_high
        self.channel_ga = channel_high
        self.channel_nl = channel_high
        self.GABlock = _GAblock(self.channel_low, self.channel_high)
        self.NLBlock = _NLblock(self.channel_ga)
        self.SABlock = _SAblock(self.channel_nl)

    def forward(self, feature_low, feature_high):
        feature_ga = self.GABlock(feature_low, feature_high)
        feature_nl = self.NLBlock(feature_ga)
        feature_sa = self.SABlock(feature_nl)
        return feature_sa


if __name__ == '__main__':

    CABlock = CAModule(256, 512)
    f_low = torch.rand((1, 256, 100, 100))
    f_high = torch.rand((1, 512, 50, 50))
    out = CABlock(f_low, f_high)
    print(out.size())

    '''sablock = SAblock(512)
    x = torch.rand((1, 512, 100, 100))
    y = sablock(x)
    print(y.size())'''

    '''nlblock = NLblock(512)
    x = torch.rand((1, 512, 100, 100))
    y = nlblock(x)
    print(y.size())'''

    '''sppblock = SPPblock()
    x1 = torch.rand((1, 10, 5, 5))
    x2 = torch.rand((1, 10, 10, 10))
    y1 = sppblock(x1)
    y2 = sppblock(x2)
    print(y1.size())
    print(y2.size())'''
    '''gablock = GAblock(512, 10, 256)
    f_low = torch.rand((1, 512, 10, 10))
    f_high = torch.rand((1, 256, 5, 5))
    y = gablock(f_low, f_high)
    print(y.size())'''
    '''
        torch.Size([1, 128, 150, 150])
        torch.Size([1, 256, 75, 75])
        torch.Size([1, 512, 38, 38])
        torch.Size([1, 1024, 19, 19])
        torch.Size([1, 512, 10, 10])
        torch.Size([1, 256, 5, 5])
        torch.Size([1, 256, 3, 3])
        '''

    '''x1 = torch.rand((1, 10, 1, 1))
    x2 = torch.rand((1, 10, 10, 10))
    y = x1 * x2
    print(y.size())
'''
