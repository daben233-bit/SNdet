import torch
from torch import nn
import torch.nn.init as init
import math
import numpy as np
# from .modules import *
from .modules import L2Norm


class Encoder(nn.Module):
    def __init__(self):
        '''
        以VGG16为基础
        conv4_3 后加了L2NORM
        fc6, fc7 被替换为conv
        然后又添加了3组卷积
        最终输出除各阶feature_low外， 还包括了第一阶的feature_high
        '''
        super(Encoder, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True) # out1
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True) # out2
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.L2Norm = L2Norm(512, 20)
        self.relu4_3 = nn.ReLU(inplace=True) # out3
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True) # out4
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.relu8_2 = nn.ReLU(inplace=True) # out5
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu9_2 = nn.ReLU(inplace=True) # out6
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.relu10_1 = nn.ReLU(inplace=True)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu10_2 = nn.ReLU(inplace=True)
        self.conv10_3 = nn.Conv2d(256, 256, kernel_size=1)# out7

        self._initialize_weights()



    def forward(self, x):
        conv1_1 = self.relu1_1(self.conv1_1(x))
        out0 = self.relu1_2(self.conv1_2(conv1_1))
        pool1 = self.pool1(out0)
        conv2_1 = self.relu2_1(self.conv2_1(pool1))
        out1 = self.relu2_2(self.conv2_2(conv2_1)) # out1
        pool2 = self.pool2(out1)
        conv3_1 = self.relu3_1(self.conv3_1(pool2))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        out2 = self.relu3_3(self.conv3_3(conv3_2)) # out2
        pool3 = self.pool3(out2)
        conv4_1 = self.relu4_1(self.conv4_1(pool3))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        #x3 = self.conv4_3(x3)
        #x3 = self.L2Norm(x3)
        #x3 = self.relu4_3(x3)
        conv4_3 = self.conv4_3(conv4_2)
        conv4_3_norm = self.L2Norm(conv4_3)
        out3 = self.relu4_3(conv4_3_norm)  # out3
        pool4 = self.pool4(out3)
        conv5_1 = self.relu5_1(self.conv5_1(pool4))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))
        pool5 = self.pool5(conv5_3)
        conv6 = self.relu6(self.conv6(pool5))
        out4 = self.relu7(self.conv7(conv6)) # out4
        conv8_1 = self.relu8_1(self.conv8_1(out4))
        out5 = self.relu8_2(self.conv8_2(conv8_1)) # out5
        conv9_1 = self.relu9_1(self.conv9_1(out5))
        out6 = self.relu9_2(self.conv9_2(conv9_1)) # out6
        conv10_1 = self.relu10_1(self.conv10_1(out6))
        conv10_2 = self.relu10_2(self.conv10_2(conv10_1))
        out7 = self.conv10_3(conv10_2) # out7


        return [out1, out2, out3, out4, out5, out6, out7]

    def _initialize_weights(self):
       for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)


if __name__ == '__main__':
    model = Encoder()
    x = torch.rand((1, 3, 768, 1024), requires_grad=True)
    #x = np.zeros((1, 3, 100, 100))
    y = model(x)
    #b = torch.nn.functional.adaptive_avg_pool2d(a, (1, 1))
    for k in y:
        print(k.size())
    '''
    with L2Norm:
        torch.Size([1, 128, 150, 150])
        torch.Size([1, 256, 75, 75])
        torch.Size([1, 512, 38, 38])
        torch.Size([1, 1024, 19, 19])
        torch.Size([1, 512, 10, 10])
        torch.Size([1, 256, 5, 5])
        torch.Size([1, 256, 3, 3])
    '''

    '''  x = np.ones((3, 3)) * 2
    y = np.ones((2, 2)) * 3
    print(x)
    print(y)
    print([x, y])'''
