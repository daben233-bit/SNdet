import sys
import os   # sys 对解释器操作（命令）的内置模块   # os 对操作系统操作（命令）的内置模块
# __file__ 为当前脚本, 形如 xxx.py
# os.path.abspath(__file__) 获取当前脚本的绝对路径（相对于执行该脚本的终端）
# os.path.dirname() 获取上级目录
# 下面嵌套了两次，即得到 父目录 的 父目录 ；同理可根据自己的需求来获取相应的目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将BASE_DIR路径添加到解释器的搜索路径列表中
sys.path.append(BASE_DIR)
from tensorboardX import SummaryWriter
from models import MyNet

import torch


def visualize_net():
    x = torch.rand((1, 3, 768, 1024))
    model = MyNet()
    with SummaryWriter(comment='MyNet') as w:
        w.add_graph(model, (x,))


if __name__ == "__main__":
    visualize_net()
