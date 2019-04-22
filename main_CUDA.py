#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from config import opt
from my_class import LWFDataset, SiameseNetwork, ContrastiveLoss, SiameseNetworkSimple


def show_plot(iteration, value):
    plt.plot(iteration, value)
    plt.show()


def train():
    # ------------------------------------ step 1/5 : 加载数据------------------------------------

    # 数据预处理设置
    mean_data_path = os.path.join(opt.lwf_data_dir, 'mean_data.txt')
    with open(mean_data_path, 'r') as f:
        lines = f.readlines()

    normMean = [float(i) for i in lines[0].split()]
    normStd = [float(i) for i in lines[1].split()]

    normTransform = transforms.Normalize(normMean, normStd)
    trainTransform = transforms.Compose([
        transforms.Resize(50),
        transforms.ToTensor(),
        normTransform
    ])

    validTransform = transforms.Compose([
        transforms.Resize(50),
        transforms.ToTensor(),
        normTransform
    ])

    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    print('load data')

    # 构建MyDataset实例
    train_data = LWFDataset(data_dir=opt.lwf_data_dir, split='pairsDevTrain', transform=trainTransform)
    valid_data = LWFDataset(data_dir=opt.lwf_data_dir, split='pairsDevTest', transform=validTransform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=opt.train_bs, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=opt.valid_bs)

    print('load data done !')

    # ------------------------------------ step 2/5 : 定义网络------------------------------------
    print('load net ')
    net = SiameseNetworkSimple().cuda()  # 创建一个网络
    print('load net done !')
    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

    criterion = ContrastiveLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=5,
                                                           verbose=True,
                                                           threshold=0.005,
                                                           threshold_mode='rel',
                                                           cooldown=0,
                                                           min_lr=0, eps=1e-08)  # 设置学习率下降策略

    # ------------------------------------ step 4/5 : 训练 --------------------------------------------------
    print('train start ')
    train_iter_index = []
    train_loss_list = []
    val_iter_index = []
    val_loss_list = []
    iteration_number = 0

    for epoch in range(opt.max_epoch):
        epoch_start_time = time.time()
        loss_sigma = 0
        net.train()  # 训练模式

        for i, data in tqdm((enumerate(train_loader))):

            # 获取图片和标签
            inputs0, inputs1, labels = data
            inputs0, inputs1, labels = Variable(inputs0).cuda(), Variable(inputs1).cuda(), Variable(labels).cuda()

            # forward, backward, update weights
            optimizer.zero_grad()
            outputs0, outputs1 = net(inputs0, inputs1)
            loss = criterion(outputs0, outputs1, labels.float())
            loss.backward()
            optimizer.step()

            iteration_number += 1
            loss_sigma += loss.item()

            if i % 10 == 0:
                print("Epoch:{},  Current loss {}".format(epoch, loss.item()))
                train_iter_index.append(iteration_number)
                train_loss_list.append(loss.item())

        # 每个epoch的 Loss, accuracy, learning rate
        lr_now = [group['lr'] for group in optimizer.param_groups][0]
        epoch_end_time = time.time()
        during_time = epoch_end_time - epoch_start_time
        loss_avg_epoch = loss_sigma / len(train_loader)
        print(
            "Training: Epoch[{:0>3}/{:0>3}] Loss_Avg_Epoch: {:.4f} Lr: {:.8f} Time: {:.2f}".format(
                epoch + 1, opt.max_epoch, loss_avg_epoch, lr_now, during_time))

        scheduler.step(loss_avg_epoch)  # 更新学习率

        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
        if epoch % 2 == 0:
            print('eval start ')
            loss_sigma = 0
            net.eval()  # 测试模式
            for i, data in tqdm((enumerate(valid_loader))):
                # 获取图片和标签
                inputs0, inputs1, labels = data
                inputs0, inputs1, labels = Variable(inputs0).cuda(), Variable(inputs1).cuda(), Variable(labels).cuda()

                # forward
                outputs0, outputs1 = net(inputs0, inputs1)
                loss = criterion(outputs0, outputs1, labels.float())

                loss_sigma += loss.item()
            val_loss_avg = loss_sigma / len(valid_loader)
            val_iter_index.append(iteration_number)
            val_loss_list.append(val_loss_avg)
            print('Epoch:{}, {} set Loss:{:.4f}'.format(epoch, 'Valid', val_loss_avg))

    print('Finished Training')

    # ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------
    plt.plot(train_iter_index, train_loss_list)
    plt.plot(val_iter_index, val_loss_list)
    plt.savefig("result.jpg")
    pass


if __name__ == '__main__':
    train()
