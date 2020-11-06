import sys
import os   # sys 对解释器操作（命令）的内置模块   # os 对操作系统操作（命令）的内置模块
# __file__ 为当前脚本, 形如 xxx.py
# os.path.abspath(__file__) 获取当前脚本的绝对路径（相对于执行该脚本的终端）
# os.path.dirname() 获取上级目录
# 下面嵌套了两次，即得到 父目录 的 父目录 ；同理可根据自己的需求来获取相应的目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将BASE_DIR路径添加到解释器的搜索路径列表中
sys.path.append(BASE_DIR)
import torch
from torch import nn
import torch.nn.init as init
import math
import numpy as np
# from models import *
import torch.nn.functional as F
# from models.encoder import Encoder
# from models.decodermodule import CAModule
from .encoder import Encoder
from .decodermodule import CAModule

class _Segblock(nn.Module):
    '''
    分割层
    '''
    def __init__(self):
        super(_Segblock, self).__init__()
        self.conv_text_pixel = nn.Conv2d(256, 1, kernel_size=1)
        self.conv_affinity_link = nn.Conv2d(256, 8, kernel_size=1)
        self.conv_repulsive_link = nn.Conv2d(256, 8, kernel_size=1)

        self._initialize_weights()
    def forward(self, x):
        '''
        text_pixel = F.softmax(self.conv_text_pixel(x), dim=1)
        affinity_link = self.conv_affinity_link(x)  # ( N 16 H W )
        affinity_link = [F.softmax(alink, dim=1) for alink in affinity_link.split(2, 1)]
        repulsive_link = self.conv_repulsive_link(x)
        repulsive_link = [F.softmax(rlink, dim=1) for rlink in repulsive_link.split(2, 1)]

        return text_pixel, torch.cat(affinity_link, 1), torch.cat(repulsive_link, 1)

        '''
        text_pixel = self.conv_text_pixel(x)
        affinity_link = self.conv_affinity_link(x)
        repulsive_link = self.conv_repulsive_link(x)
        
        return text_pixel, affinity_link, repulsive_link
    
    def _initialize_weights(self):
       for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)


class MyNet(nn.Module):
    '''
        encoder各阶输出size：
        torch.Size([1, 128, 150, 150])
        torch.Size([1, 256, 75, 75])
        torch.Size([1, 512, 38, 38])
        torch.Size([1, 1024, 19, 19])
        torch.Size([1, 512, 10, 10])
        torch.Size([1, 256, 5, 5])
        torch.Size([1, 256, 3, 3])
    '''
    def __init__(self):
        super(MyNet, self).__init__()
        self.encoder = Encoder()
        self.decoder1 = CAModule(256, 256)
        self.decoder2 = CAModule(512, 256)
        self.decoder3 = CAModule(1024, 256)
        self.decoder4 = CAModule(512, 256)
        self.decoder5 = CAModule(256, 256)
        self.decoder6 = CAModule(128, 256)
        self.seg1 = _Segblock()
        self.seg2 = _Segblock()
        self.seg3 = _Segblock()
        self.seg4 = _Segblock()
        self.seg5 = _Segblock()
        self.seg_upsample1 = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)
        self.seg_upsample2 = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)
        self.seg_upsample3 = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)
        self.seg_upsample4 = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1)

        self.affinity_upsample1 = nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)
        self.affinity_upsample2 = nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)
        self.affinity_upsample3 = nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)
        self.affinity_upsample4 = nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)

        self.repulsive_upsample1 = nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)
        self.repulsive_upsample2 = nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)
        self.repulsive_upsample3 = nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)
        self.repulsive_upsample4 = nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)

        self.sigmoid_layer = nn.Sigmoid()


    def forward(self, x):
        feature_low1, feature_low2, feature_low3, feature_low4, feature_low5, feature_low6, feature_high1 = self.encoder(x)
        feature_high2 = self.decoder1(feature_low6, feature_high1)
        feature_high3 = self.decoder2(feature_low5, feature_high2)
        feature_high4 = self.decoder3(feature_low4, feature_high3)
        feature_high5 = self.decoder4(feature_low3, feature_high4)
        feature_high6 = self.decoder5(feature_low2, feature_high5)
        feature_high7 = self.decoder6(feature_low1, feature_high6)
        seg1, affinity1, repulsive1 = self.seg1(feature_high3)
        seg2, affinity2, repulsive2 = self.seg2(feature_high4)
        seg3, affinity3, repulsive3 = self.seg3(feature_high5)
        seg4, affinity4, repulsive4 = self.seg4(feature_high6)
        seg5, affinity5, repulsive5 = self.seg5(feature_high7)
        
        seg_upsample1 = self.seg_upsample1(seg1) + seg2
        seg_upsample2 = self.seg_upsample2(seg_upsample1) + seg3
        seg_upsample3 = self.seg_upsample3(seg_upsample2) + seg4
        seg_upsample4 = self.seg_upsample4(seg_upsample3) + seg5

        affinity_upsample1 = self.affinity_upsample1(affinity1) + affinity2
        affinity_upsample2 = self.affinity_upsample2(affinity_upsample1) + affinity3
        affinity_upsample3 = self.affinity_upsample3(affinity_upsample2) + affinity4
        affinity_upsample4 = self.affinity_upsample4(affinity_upsample3) + affinity5

        repulsive_upsample1 = self.repulsive_upsample1(repulsive1) + repulsive2
        repulsive_upsample2 = self.repulsive_upsample2(repulsive_upsample1) + repulsive3
        repulsive_upsample3 = self.repulsive_upsample3(repulsive_upsample2) + repulsive4
        repulsive_upsample4 = self.repulsive_upsample4(repulsive_upsample3) + repulsive5

        seg_upsample4 = self.sigmoid_layer(seg_upsample4)
        affinity_upsample4 = self.sigmoid_layer(affinity_upsample4)
        repulsive_upsample4 = self.sigmoid_layer(repulsive_upsample4)

        return seg_upsample4, affinity_upsample4, repulsive_upsample4


if __name__ == '__main__':
    '''segblock = _Segblock()
    x = torch.rand((1, 256, 100, 100))
    y = segblock(x)
    print(yi.size() for yi in y)'''
    '''x1 = torch.ones((1, 256, 100, 100))
    x2 = torch.ones((1, 256, 100, 100)) * 5
    y = (x1 + x2) / 2
    print(y.size())'''
    x = torch.rand((1, 3, 768, 1024))
    net = MyNet()
    y1, y2, y3 = net(x)
    print(y1.size())
    # print(y1)
    print(y2.size())
    # print(y2)
    # print(y3)
    print(y3.size())
    '''
    torch.Size([1, 2, 150, 150])
    torch.Size([1, 16, 150, 150])
    torch.Size([1, 16, 150, 150])
    '''
    # x = torch.rand((1, 3, 300, 300))
    # x1 = F.interpolate(x, size=150)
    # print(x1.shape)
