'''
将polygon标签转换为segmentation mask
'''
import sys
import os   # sys 对解释器操作（命令）的内置模块   # os 对操作系统操作（命令）的内置模块
# __file__ 为当前脚本, 形如 xxx.py
# os.path.abspath(__file__) 获取当前脚本的绝对路径（相对于执行该脚本的终端）
# os.path.dirname() 获取上级目录
# 下面嵌套了两次，即得到 父目录 的 父目录 ；同理可根据自己的需求来获取相应的目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将BASE_DIR路径添加到解释器的搜索路径列表中
sys.path.append(BASE_DIR)
import torchvision
from config import sncfg as cfg

from skimage import draw, io
import numpy as np
import cv2
import os
import sys
import cv2
import torch
import torch.nn.functional as F


np.set_printoptions(threshold=np.inf)

list_root = '/home/luokun/data/datasets/TDGYB/'
train_list = 'train.txt'
test_list = 'test.txt'


def poly2mask(blobs, h, w, shrink_ratio, cropped_cor_top_left=None):
    '''
    :param blobs: polygon集合， [ [各点归一化x轴坐标], [各点归一化y轴坐标], [原始图像高度、宽度] ]
    :param path_to_masks_folder:保存mask的文件夹
    :param h: mask height = 1/2 image height
    :param w: mask width
    :param shrink_ratio: repulsive link mask shrink ratio
    :return:
    '''
    mask = np.zeros((h, w), dtype=np.int)
    # print('mask: ' + str(mask.shape))
    mask_weight = np.zeros((h, w), dtype=np.float)
    # print('mask_weight: ' + str(mask_weight.shape))
    mask_instance_seperate = []
    mask_shrink = np.zeros((h, w), dtype=np.int)
    for l in blobs:
        l_x_norm = l[0]
        l_y_norm = l[1]
        l_x_rescale = [int(x * w) for x in l_x_norm]
        l_y_rescale = [int(y * h) for y in l_y_norm]
        # print(l_x_rescale)

        # print(l_x_rescale)
        x_ave = np.sum(l_x_rescale) / 4
        y_ave = np.sum(l_y_rescale) / 4
        l_x_shrink = np.asarray(x_ave - shrink_ratio * (x_ave - l_x_rescale)).astype(np.int)
        l_y_shrink = np.asarray(y_ave - shrink_ratio * (y_ave - l_y_rescale)).astype(np.int)
        mask_i = np.zeros((h, w), dtype=np.int)
        mask_shrink_i = np.zeros((h, w), dtype=np.int)
        fill_col_coords, fill_row_coords = draw.polygon(l_y_rescale, l_x_rescale, l[2])
        fill_col_co_shrink, fill_row_co_shrink = draw.polygon(l_y_shrink, l_x_shrink, l[2])
        # print(fill_row_coords)
        # print(fill_col_coords)
        mask_i[fill_col_coords, fill_row_coords] = 1
        mask_shrink_i[fill_col_co_shrink, fill_row_co_shrink] = 1
        mask_instance_seperate.append(mask_i)
        mask_shrink += mask_shrink_i
        # print(mask_i)
        mask += mask_i
    mask_not_overlapped = mask == 1  # get final mask
    mask_shrink_not_over_lapped = mask_shrink == 1
    mask_shrink_not_over_lapped_reverse = 1 - mask_shrink_not_over_lapped
    # io.imsave('E:/dataset/TD/temp/reverse.png', mask_shrink_not_over_lapped_reverse.astype(np.int) * 255)
    mask_repulsive = mask_not_overlapped * mask_shrink_not_over_lapped_reverse
    # io.imsave('E:/dataset/TD/temp/mask_repulsive.png', mask_repulsive.astype(np.int) * 255)
    # print('mask_not_overlapped: ' + str(mask_not_overlapped.shape))
    # io.imsave(path_to_masks_folder, mask_not_overlapped)
    pixel_repulsive = np.zeros((h, w), dtype=np.int)
    for i, mask_i in enumerate(mask_instance_seperate):
        w_i = np.sum(mask_instance_seperate) / len(mask_instance_seperate) / np.sum(mask_i)
        mask_weight += mask_i * w_i  # get final mask weight
        mask_i *= i + 1
        pixel_repulsive += mask_i
    pixel_repulsive *= mask_not_overlapped
    # io.imsave('E:/dataset/TD/temp/pixel_repulsive.png', pixel_repulsive)
    pixel_link = np.zeros((h, w, 8), dtype=np.int)
    pixel_repulsive_link = np.zeros((h, w, 8), dtype=np.int)
    pixel_repulsive_link_weight = np.zeros((h, w, 8), dtype=np.int)
    # print('pixel_link: ' + str(pixel_link.shape))
    mask_positive_points_coords = np.where(mask_not_overlapped == 1)
    repulsive_points_set = np.where(mask_repulsive)
    pixel_link[mask_positive_points_coords] = 1
    # print(mask_not_overlapped)
    mask_copy = mask_not_overlapped.copy().astype(np.uint8)
    # print('mask_copy: ' + str(mask_copy.shape))
    # print(type(mask_copy))  # <class 'numpy.ndarray'>
    # print(mask_copy.dtype)  # uint8
    border_set, _ = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(border_set)
    # cv2.drawContours(mask_copy, border_set, -1, color=1, thickness=3)
    def in_bbox(x, y):
        return mask_not_overlapped[x, y]

    for border in border_set:
        # print('=================================')
        # print(border.shape)  # (12, 1, 2)
        # print(len(border))  # 12
        # print(type(border))   # <class 'numpy.ndarray'>
        # print(border)

        for point in border:
            y, x = np.squeeze(point, axis=0)
            neighbours = find_neighbours(x, y)
            for idx, (x_i, y_i) in enumerate(neighbours):
                if not is_valid_cord(x_i, y_i, w, h) or not in_bbox(x_i, y_i):
                    pixel_link[x, y, idx] = 0  # get final pixel link
    mask_weight = np.asarray(mask_weight, dtype=np.float)
    # print('mask_weight: ' + str(mask_weight.shape))
    pixel_link_weight = np.ones((h, w, 8), dtype=np.float)
    pixel_link_weight *= np.expand_dims(mask_weight, axis=-1)  # get final link weight
    # print('pixel_link_weight: ' + str(pixel_link_weight.shape))
    pixel_link = pixel_link.transpose((2, 0, 1))
    # test_border1 = border_set[0]
    # test_point = test_border1[0]
    # test_y, test_x = np.squeeze(test_point, axis=0)
    # for pl in pixel_link:
    #     print(pl[test_x, test_y])
    pixel_link_weight = pixel_link_weight.transpose((2, 0, 1))
    repulsive_points_set = np.asarray(repulsive_points_set)
    # print(len(repulsive_points_set))
    # print(type(repulsive_points_set))
    # print(repulsive_points_set.shape)
    xs, ys = repulsive_points_set
    # print(xs)
    # print(w)
    # print(ys)
    # print(h)
    for x, y in zip(xs, ys):
        neighbours_repulsive = find_neighbours(x, y)
        for idx, (x_i, y_i) in enumerate(neighbours_repulsive):
            if pixel_repulsive[x, y] != pixel_repulsive[x_i, y_i] and is_valid_cord(x_i, y_i, h, w):  # 0--4000 0--6000
                pixel_repulsive_link[x, y, idx] = 1  # get final repulsive link
                if pixel_repulsive[x_i, y_i] != 0:
                    pixel_repulsive_link_weight[x, y, idx] = 2
                else:
                    pixel_repulsive_link_weight[x, y, idx] = 1  # get final repulsive link weight

    pixel_repulsive_link = pixel_repulsive_link.transpose((2, 0, 1))
    pixel_repulsive_link_weight = pixel_repulsive_link_weight.transpose((2, 0, 1))
    # test_set = np.where(pixel_repulsive_link == 1)
    # print(test_set)

    if cropped_cor_top_left:
        img_crop_h, img_crop_w = cfg.crop_size
        crop_h = int(img_crop_h / 2)
        crop_w = int(img_crop_w / 2)
        x_crop_top_left_norm, y_crop_top_left_norm = cropped_cor_top_left
        x_crop_top_left = int(x_crop_top_left_norm * w)
        y_crop_top_left = int(y_crop_top_left_norm * h)

        mask_not_overlapped_cropped = mask_not_overlapped[y_crop_top_left:y_crop_top_left +
                                                          crop_h, x_crop_top_left:x_crop_top_left + crop_w]
        mask_weight_cropped = mask_weight[y_crop_top_left:y_crop_top_left +
                                          crop_h, x_crop_top_left:x_crop_top_left + crop_w]
        pixel_link_cropped = pixel_link[:, y_crop_top_left:y_crop_top_left +
                                        crop_h, x_crop_top_left:x_crop_top_left + crop_w]
        pixel_link_weight_cropped = pixel_link_weight[:,
                                                      y_crop_top_left:y_crop_top_left + crop_h, x_crop_top_left:x_crop_top_left + crop_w]
        pixel_repulsive_link_cropped = pixel_repulsive_link[:,
                                                            y_crop_top_left:y_crop_top_left + crop_h, x_crop_top_left:x_crop_top_left + crop_w]
        pixel_repulsive_link_weight_cropped = pixel_repulsive_link_weight[:,
                                                                          y_crop_top_left: y_crop_top_left + crop_h, x_crop_top_left: x_crop_top_left + crop_w]
        mask_repulsive_cropped = mask_repulsive[y_crop_top_left: y_crop_top_left +
                                                crop_h, x_crop_top_left: x_crop_top_left + crop_w]

        return np.expand_dims(mask_not_overlapped_cropped, axis=0), np.expand_dims(mask_weight_cropped, axis=0), pixel_link_cropped, pixel_link_weight_cropped, \
            pixel_repulsive_link_cropped, pixel_repulsive_link_weight_cropped, np.expand_dims(mask_repulsive_cropped, axis=0)

    return np.expand_dims(mask_not_overlapped, axis=0), np.expand_dims(mask_weight, axis=0), pixel_link, pixel_link_weight, \
        pixel_repulsive_link, pixel_repulsive_link_weight, np.expand_dims(mask_repulsive, axis=0)


def find_neighbours(x, y):
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
            (x - 1, y),                 (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >= 0 and x < w and y >= 0 and y < h


def main():
    fileslist = []
    with open(list_root + train_list, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            img, gt = line.split(' ')
            fileslist.append([img, gt])

    with open(list_root + test_list, 'r') as f2:
        lines = f2.readlines()
        for line in lines:
            img, gt = line.split(' ')
            fileslist.append([img, gt])

    f1.close()
    f2.close()

    for img, gt in fileslist:
        image = io.imread(img)  # H W C
        H, W, _ = image.shape
        shape = (int(H), int(W))
        blobs = []
        # l = []

        with open(gt.replace('\n', ''), 'r', encoding='utf-8') as f3:
            lines = f3.readlines()
            for line in lines:
                l = []
                x1, y1, x2, y2, x3, y3, x4, y4 = line.split(',')[:8]
                l.append([int(float(x1)), int(float(x2)), int(float(x3)), int(float(x4))])
                l.append([int(float(y1)), int(float(y2)), int(float(y3)), int(float(y4))])
                l.append(shape)  # l: [[xs coords], [ys coords], [img h and w]]
                blobs.append(l)
        poly2mask(blobs, img.replace('IMAGES', 'MASKS'), H, W)
        f3.close()


def test():
    test_path = 'E:/dataset/TD/temp/'
    img = '39.jpg'
    gt = '39.txt'
    image = io.imread(test_path + img)
    h, w, _ = image.shape
    shape = (int(h), int(w))
    blobs = []
    with open(test_path + gt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            l = []
            x1, y1, x2, y2, x3, y3, x4, y4 = line.split(',')[:8]
            l.append([int(float(x1)), int(float(x2)), int(float(x3)), int(float(x4))])
            l.append([int(float(y1)), int(float(y2)), int(float(y3)), int(float(y4))])
            l.append(shape)
            blobs.append(l)
    poly2mask(blobs, test_path + '39.png', h, w)
    f.close()


def testcv2():
    test_path = 'E:/dataset/TD/temp/'
    img = '39.jpg'
    gt = '39.png'
    image = cv2.imread(test_path + img)
    print(image.shape)
    mask = cv2.imread(test_path + gt)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    print(type(mask))  # <class 'numpy.ndarray'>
    print(mask.shape)
    print(mask.dtype)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # print(hierarchy)
    cv2.drawContours(image, contours, -1, color=(0, 0, 255), thickness=3)
    cv2.imwrite(test_path + 'temp.jpg', image)
    # cv2.imshow('mask', image)
    # cv2.waitKey(0)


def testcoords():
    mat = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    print(mat)
    coords = np.where(mat == 1)
    print(coords)  # (array([1, 1, 2], dtype=int64), array([1, 2, 2], dtype=int64))


def testp2m():
    test_path = '/home/luokun/data/datasets/TDGYB/temp/'
    img = '39.jpg'
    gt = '39.txt'
    image = cv2.imread(test_path + img)
    shrink_ratio = 0.89
    # print(image.shape)
    # image = image.transpose((2, 0, 1))
    # print(image.shape) # 4000, 6000, 3
    h, w, _ = image.shape
    # print(h) # 4000
    shape = [h, w]
    blobs = []
    with open(test_path + gt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            l = []
            x1, y1, x2, y2, x3, y3, x4, y4 = line.split(',')[:8]
            l.append([int(float(x1)), int(float(x2)), int(float(x3)), int(float(x4))])
            l.append([int(float(y1)), int(float(y2)), int(float(y3)), int(float(y4))])
            l.append(shape)
            blobs.append(l)
    f.close()
    mask, mask_weight, links, link_weights, repulsives, repulsives_weights, mask_repulsive = poly2mask(blobs, h, w, shrink_ratio)
    # print(len(np.where(mask == 1)))
    # positive_set = np.where(mask == 1)
    # print(mask_weight[positive_set]
    print(mask.shape)  # (4000, 6000)
    print(mask_weight.shape)  # (4000, 6000)
    print(links.shape)  # (8, 4000, 6000)
    print(link_weights.shape)  # (8, 4000, 6000)
    print(repulsives.shape)
    print(repulsives_weights.shape)
    print(mask_repulsive.shape)
    '''
    torch.Size([1, 2000, 3000])
    torch.Size([1, 2000, 3000])
    torch.Size([8, 2000, 3000])
    torch.Size([8, 2000, 3000])
    torch.Size([8, 2000, 3000])
    torch.Size([8, 2000, 3000])
    '''
    # io.imsave(test_path + 'mask.png', mask.squeeze(0).squeeze(0) * 255)
    # io.imsave(test_path + 'mask_weight.png', mask_weight.squeeze(0).squeeze(0) * 255)
    # for i, link in enumerate(links.squeeze(0)):
    #     io.imsave(test_path + 'link' + f'{i}.png', link * 255)
    # for i, link_weight in enumerate(link_weights.squeeze(0)):
    #     io.imsave(test_path + 'link_weight' + f'{i}.png', link_weight * 255)
    # for i, repulsive in enumerate(repulsives.squeeze(0)):
    #     io.imsave(test_path + 'repulsive' + f'{i}.png', repulsive * 255)
    # for i, repulsive_weight in enumerate(repulsives_weights.squeeze(0)):
    #     io.imsave(test_path + 'repulsive_weight' + f'{i}.png', repulsive_weight * 122)


def test2():
    test_path = 'E:/dataset/TD/temp/'
    img = '39.jpg'
    gt = '39.txt'
    image = io.imread(test_path + img)
    print(image.shape)


if __name__ == '__main__':
    # main()
    # test()
    # testcv2()
    # testcoords()
    testp2m()
    # test2()
    # x = torch.rand((1, 3, 400, 300))
    # x1 = F.interpolate(x, size=(200, 150))
    # print(x1.size())
