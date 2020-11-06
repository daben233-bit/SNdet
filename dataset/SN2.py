import sys
import os   # sys 对解释器操作（命令）的内置模块   # os 对操作系统操作（命令）的内置模块
# __file__ 为当前脚本, 形如 xxx.py
# os.path.abspath(__file__) 获取当前脚本的绝对路径（相对于执行该脚本的终端）
# os.path.dirname() 获取上级目录
# 下面嵌套了两次，即得到 父目录 的 父目录 ；同理可根据自己的需求来获取相应的目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将BASE_DIR路径添加到解释器的搜索路径列表中
sys.path.append(BASE_DIR)
from skimage import io
import cv2
import numpy as np
from torch.utils import data
from torchvision import transforms as T
import util
import torch
from config import sncfg as cfg


class SN(data.Dataset):
    def __init__(self, root, transform=None, train=True, task='detection', shrink_ratio=None, crop_size=None):
        self.root = root
        self.train_list = 'train.txt'
        self.test_list = 'test.txt'
        self.transform = transform
        self.train = train
        self.task = task
        self.crop_size = crop_size
        if shrink_ratio == None:
            self.shrink_ratio = 0.89
        if self.train:
            with open(self.root + self.train_list, 'r', encoding='utf-8') as f:
                data = f.readlines()
                data = [line.strip().split(' ') for line in data]
        else:
            with open(self.root + self.test_list, 'r', encoding='utf-8') as f:
                data = f.readlines()
                data = [line.strip().split(' ') for line in data]
        f.close()
        self.data = data
        if self.task == 'detection':
            if self.transform == None:
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                if self.train:
                    self.transforms = T.Compose([
                        T.ToTensor(),
                        normalize
                    ])
                else:
                    self.transforms = T.Compose([
                        T.ToTensor(),
                        normalize
                    ])
        else:
            raise Exception('目前只支持检测任务')

    def __getitem__(self, index):
        img_path, gt_path = self.data[index]
        image = cv2.imread(img_path)
        if self.task == 'detection':
            h, w, _ = image.shape
            # print(h)  # 4000
            # image = image.transpose((2, 0, 1))
            image = self.transforms(image).double()
            # image = image.transpose((2, 0, 1))
            shape = [h, w]
            blobs = []
            with open(gt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    l = []
                    x1, y1, x2, y2, x3, y3, x4, y4 = line.split(',')[:8]
                    l.append([int(float(x1)), int(float(x2)), int(float(x3)), int(float(x4))])
                    l.append([int(float(y1)), int(float(y2)), int(float(y3)), int(float(y4))])
                    l.append(shape)
                    blobs.append(l)
            if self.crop_size:
                import random
                crop_h, crop_w = self.crop_size
                blob = blobs[random.randint(0, len(blobs) - 1)]
                x_max = np.max(blob[0])
                x_min = np.min(blob[0])
                y_max = np.max(blob[1])
                y_min = np.min(blob[1])
                row_min = np.max((0, x_max - crop_w))
                row_max = np.min((x_min, w - crop_w))
                col_min = np.max((0, y_max - crop_h))
                col_max = np.min((y_min, h - crop_h))
                x_crop_top_left = random.randint(row_min, row_max)
                y_crop_top_left = random.randint(col_min, col_max)
                blobs_cropped = []
                for l in blobs:
                    l_cropped = []
                    x_cor_group_cropped = [x - x_crop_top_left for x in l[0]]
                    l_cropped.append(x_cor_group_cropped)
                    y_cor_group_cropped = [y - y_crop_top_left for y in l[1]]
                    l_cropped.append(y_cor_group_cropped)
                    l_cropped.append(self.crop_size)
                    blobs_cropped.append(l_cropped)
                print(blobs_cropped)
                image_cropped = image[:, y_crop_top_left:y_crop_top_left +
                                      crop_h, x_crop_top_left:x_crop_top_left + crop_w]
                mask_cropped, mask_weight_cropped, links_cropped, link_weights_cropped, repulsives_cropped, repulsives_weights_cropped, mask_repulsive_cropped = util.poly2mask(blobs_cropped, crop_h, crop_w,
                                                                                                                                                                                self.shrink_ratio)
                return image_cropped, [mask_cropped > 0, mask_weight_cropped, links_cropped > 0, link_weights_cropped, repulsives_cropped > 0, repulsives_weights_cropped, mask_repulsive_cropped > 0]
            
            
            mask, mask_weight, links, link_weights, repulsives, repulsives_weights, mask_repulsive = util.poly2mask(blobs, h, w,
                                                                                                                    self.shrink_ratio, self.crop_size)
            return image, [mask > 0, mask_weight, links > 0, link_weights, repulsives > 0, repulsives_weights, mask_repulsive > 0]
        else:
            raise Exception('目前只支持检测任务')

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    sn = SN(root='/home/luokun/data/datasets/TDGYB/', crop_size=cfg.crop_size)
    save_root = '/home/luokun/data/datasets/TDGYB//temp2/'
    image, [mask, mask_weight, links, link_weights, repulsives, repulsives_weights, mask_repulsive] = sn.__getitem__(600)
    print(image.size())
    print(mask.size())
    print(mask)
    print(mask_weight.size())
    print(mask_weight)
    print(links.size())
    print(link_weights.size())
    print(repulsives.size())
    print(repulsives_weights.size())
    print(mask_repulsive.size())

    mi = np.nanmin(mask_weight)
    print(mi)

    image = image.numpy()

    io.imsave(save_root + 'item.jpg', np.transpose(image, (1, 2, 0)))
    mask = mask.numpy()
    io.imsave(save_root + 'mask.png', mask.squeeze(0))
    mask_weight = mask_weight.numpy()
    io.imsave(save_root + 'mask_weight.png', mask_weight.squeeze(0))
    for i, link in enumerate(links):
        io.imsave(save_root + 'link' + f'{i}.png', link.numpy())
    for i, link_weight in enumerate(link_weights):
        io.imsave(save_root + 'link_weight' + f'{i}.png', link_weight.numpy())
    for i, repulsive in enumerate(repulsives):
        io.imsave(save_root + 'repulsive' + f'{i}.png', repulsive.numpy())
    for i, repulsive_weight in enumerate(repulsives_weights):
        io.imsave(save_root + 'repulsive_weight' + f'{i}.png', repulsive_weight.numpy() / 2)
    io.imsave(save_root + 'mask_repulsive' + f'{i}.png', mask_repulsive.numpy().squeeze(0))

    # for part in item:
    #     print(part.shape)
    #     '''
    #     (3, 4000, 6000)
    #     (4000, 6000)
    #     (4000, 6000)
    #     (8, 4000, 6000)
    #     (8, 4000, 6000)
    #     (8, 4000, 6000)
    #     (8, 4000, 6000)
    #     '''
