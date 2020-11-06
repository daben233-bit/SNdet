import os
import random

rootdir = '/home/luokun/data/datasets/TDGYB/'
imgdir = '/IMAGES/'
gtdir = '/TXTS/'
total = 1296
ratio = 0.7

def generatefilelist():
    fileslist = []
    for root, dirs, imgs in os.walk(rootdir + imgdir):
        for img in imgs:
            imgroot = root + img
            gtroot = root.replace('IMAGES', 'TXTS') + img.replace('.jpg', '.txt')
            file = [imgroot, gtroot]
            fileslist.append(file)
    random.shuffle(fileslist)
    offset = int(total * ratio)
    trainfileslist = fileslist[:offset]
    testfileslist = fileslist[offset:]
    with open(rootdir + 'train.txt', 'w') as f_train:
        for img, gt in trainfileslist:
            f_train.write(img + ' ' + gt + '\n')

    with open(rootdir + 'test.txt', 'w') as f_test:
        for img, gt in testfileslist:
            f_test.write(img + ' ' + gt + '\n')




if __name__ == '__main__':
    generatefilelist()