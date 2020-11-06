import sys
import os   # sys 对解释器操作（命令）的内置模块   # os 对操作系统操作（命令）的内置模块
# __file__ 为当前脚本, 形如 xxx.py
# os.path.abspath(__file__) 获取当前脚本的绝对路径（相对于执行该脚本的终端）
# os.path.dirname() 获取上级目录
# 下面嵌套了两次，即得到 父目录 的 父目录 ；同理可根据自己的需求来获取相应的目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将BASE_DIR路径添加到解释器的搜索路径列表中
sys.path.append(BASE_DIR)
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
from dataset import SN
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import MyNet
from util import criterion, BalancedDataParallel
import torch
import torch.nn.parallel as parallel 
from torch.autograd import Variable
from config import sncfg as cfg
from torchnet import meter

torch.autograd.set_detect_anomaly(True)
def train(**kwargs):
    # 1. configure model
    cfg._parse(kwargs)
    model = MyNet()
    if cfg.load_model_path:
        model.load_state_dict(torch.load(cfg.load_model_path))

    if cfg.multi_gpu:
        model = parallel.DataParallel(model)
    
    if cfg.use_gpu:
        model.cuda()
    
    
    # 2. prepare data
    train_data = SN(root=cfg.train_data_root, crop_size=cfg.crop_size)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)

    # 3. criterion (already imported) and optimizer
    lr = cfg.lr
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum)

    # 4. meters
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10

    # train
    for epoch in range(cfg.max_epoch):
        print('epoch %s: ===========================' % epoch)
        loss_meter.reset()

        for ii, (data, label_group) in tqdm(enumerate(train_loader)):
            # train model
            if cfg.use_gpu:
                data = data.cuda()
                label_group = [label.cuda() for label in label_group]
            data = Variable(data).float()
            label_group = [Variable(label) for label in label_group]
           
            optimizer.zero_grad()
            score = model(data)
            # for item in score:
            #     print(item)
            loss = criterion(score, label_group, batch_size=cfg.batch_size, neg_pos_ratio=cfg.neg_pos_ratio)
            loss.backward()
            optimizer.step()

            # meters update and print
            loss_meter.add(loss.item())
            if (ii + 1) % cfg.print_freq == 0:
                print(loss_meter.value()[0])
        
        if (epoch + 1) % cfg.save_freq == 0:
            torch.save(model.module.state_dict(), f'./checkpoints/last.pth')
        
        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * cfg.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]






if __name__ == '__main__':
    train()
