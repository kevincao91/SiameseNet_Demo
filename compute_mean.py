# coding: utf-8

import numpy as np
import cv2
import os

"""
    对train-set和valid-set所有图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""

train_txt_path = '/home/kevin/文档/LFW/peopleDevTrain.txt'
mean_data_path = '/home/kevin/文档/LFW/mean_data.txt'
img_dir = '/home/kevin/文档/LFW/lfw/'

Max_Num = 200  # 最多多少图片进行计算  0表示不限制  4039

img_h, img_w = 250, 250
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(train_txt_path, 'r') as f:
    lines_train = f.readlines()

lines = lines_train[1:-1]

index = 0
for line in lines:
    words = line.rstrip().split()
    img_path = os.path.join(img_dir, words[0], words[0] + "_%0*d.jpg" % (4, int(words[1])))
    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_h, img_w))

    img = img[:, :, :, np.newaxis]
    imgs = np.concatenate((imgs, img), axis=3)
    if (index >= Max_Num) and (Max_Num != 0):
        break
    index += 1
    print(index)


imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse()  # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

with open(mean_data_path, 'w') as f:
    f.write(str(means[0])+' ')
    f.write(str(means[1])+' ')
    f.write(str(means[2])+'\n')
    f.write(str(stdevs[0])+' ')
    f.write(str(stdevs[1])+' ')
    f.write(str(stdevs[2]))
