#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from config import opt
from my_class import LWFDataset, SiameseNetwork, ContrastiveLoss, SiameseNetworkWED


def show_plot(train_iter_index, train_loss_list, val_iter_index, val_loss_list, val_acc_list):
    plt.figure('Loss and Accuracy')
    plt.subplot(1, 2, 1)
    plt.plot(train_iter_index, train_loss_list)
    plt.plot(val_iter_index, val_loss_list)
    plt.subplot(1, 2, 2)
    plt.plot(val_iter_index, val_acc_list)
    plt.ioff()
    plt.show()


def cal_best_m(pos_num, pos_avg, neg_avg, euclidean_distance_list):
    n = 100
    d = (neg_avg - pos_avg) / n
    right_list = []
    for i in range(n):
        m = pos_avg + i * d
        right_num = 0
        for euclidean_distance in euclidean_distance_list[1:pos_num]:
            if euclidean_distance < m:
                right_num += 1
        for euclidean_distance in euclidean_distance_list[pos_num+1:-1]:
            if euclidean_distance > m:
                right_num += 1
        acc = right_num / (pos_num * 2)
        right_list.append(acc)

    best_acc = max(right_list)
    best_index = right_list.index(best_acc)
    best_m = pos_avg + best_index * d
    return best_m, best_acc


def show_distance(euclidean_distance_list):
    valid_data_num = len(euclidean_distance_list)
    x = range(valid_data_num)
    pos_num = valid_data_num / 2
    pos_num = int(pos_num)
    pos_avg = np.average(euclidean_distance_list[1:pos_num])
    neg_avg = np.average(euclidean_distance_list[pos_num + 1:-1])
    best_m, best_acc = cal_best_m(pos_num, pos_avg, neg_avg, euclidean_distance_list)
    plt.figure('euclidean distance distribute')
    plt.scatter(x, euclidean_distance_list, s=36, alpha=0.5)  # s为size，按每个点的坐标绘制，alpha为透明度
    plt.plot((0, pos_num - 1), (pos_avg, pos_avg), 'b')
    plt.text(pos_num / 2, pos_avg+0.1, format(pos_avg, '0.2f'), fontsize=9)
    plt.plot((pos_num, pos_num * 2), (neg_avg, neg_avg), 'r')
    plt.text(pos_num * 1.5, neg_avg+0.1, format(neg_avg, '0.2f'), fontsize=9)
    plt.plot((0, pos_num * 2), (best_m, best_m), 'g')
    plt.text(pos_num / 2, best_m + 0.1, 'best m:%s, best_acc:%s' % (format(best_m, '0.2f'), format(best_acc, '0.2f')),
             fontsize=9)
    plt.ioff()
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
        transforms.Resize(100),
        transforms.ToTensor(),
        normTransform
    ])

    validTransform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        normTransform
    ])

    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    # print('load data')

    # 构建MyDataset实例
    train_data = LWFDataset(data_dir=opt.lwf_data_dir, split='pairsDevTrain', transform=trainTransform)
    # train_data = LWFDataset(data_dir=opt.lwf_data_dir, split='pairsDevTest', transform=trainTransform)
    valid_data = LWFDataset(data_dir=opt.lwf_data_dir, split='pairsDevTest', transform=validTransform)
    train_data_num = len(train_data)
    valid_data_num = len(valid_data)
    print(train_data_num, valid_data_num)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=opt.train_bs, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=opt.valid_bs)

    print('load data done !')

    # ------------------------------------ step 2/5 : 定义网络------------------------------------
    # print('load net ')
    net = SiameseNetworkWED()  # 创建一个网络
    print('load net done !')
    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

    criterion = torch.nn.BCEWithLogitsLoss()
    lr_init = 0.001
    # optimizer = optim.Adam(net.parameters(), lr=0.0005)
    optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1, weight_decay=0.0001)  # 选择优化器
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
    val_acc_list = []
    iteration_number = 0

    for epoch in range(opt.max_epoch):
        loss_sigma = 0
        net.train()  # 训练模式

        for i, data in tqdm((enumerate(train_loader))):

            # 获取图片和标签
            inputs0, inputs1, labels = data
            inputs0, inputs1, labels = Variable(inputs0), Variable(inputs1), Variable(labels.float())

            # forward, backward, update weights
            optimizer.zero_grad()
            outputs = net(inputs0, inputs1)
            labels = labels.view((outputs.size()[0], -1))
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iteration_number += 1
            loss_sigma += loss.item()

            if i % 10 == 0:
                # print("Epoch:{},  Current loss {}".format(epoch, loss.item()))
                train_iter_index.append(iteration_number)
                train_loss_list.append(loss.item())

        # 每个epoch的 Loss, accuracy, learning rate
        lr_now = [group['lr'] for group in optimizer.param_groups][0]
        loss_avg_epoch = loss_sigma / len(train_loader)
        print(
            "Training: Epoch[{:0>3}/{:0>3}] Loss_Avg_Epoch: {:.4f}       Lr: {:.8f}".format(
                epoch + 1, opt.max_epoch, loss_avg_epoch, lr_now))

        scheduler.step(loss_avg_epoch)  # 更新学习率

        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
        if epoch % 1 == 0:
            # print('eval start ')
            loss_sigma = 0
            cls_num = 2
            conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
            euclidean_distance_list = []
            net.eval()  # 测试模式
            for i, data in tqdm((enumerate(valid_loader))):
                # 获取图片和标签
                inputs0, inputs1, labels = data
                inputs0, inputs1, labels = Variable(inputs0), Variable(inputs1), Variable(labels.float())

                # forward
                outputs = net(inputs0, inputs1)
                labels = labels.view((outputs.size()[0], -1))
                loss = criterion(outputs, labels)

                loss_sigma += loss.item()

                # 统计
                euclidean_distance = torch.sigmoid(outputs)
                # print(euclidean_distance.data)
                predicted = []
                for tt in range(len(euclidean_distance)):
                    euclidean_distance_list.append(euclidean_distance[tt].item())
                    if euclidean_distance[tt] < 0.5:
                        predicted.append(0)  # positive pairs
                    else:
                        predicted.append(1)   # negative pairs
                # print(predicted, labels.data)
                # 统计混淆矩阵
                for j in range(len(labels)):
                    gt_i = int(labels[j].numpy())
                    pre_i = predicted[j]
                    conf_mat[gt_i, pre_i] += 1.0

            # print(conf_mat)
            val_acc_avg = conf_mat.trace() / conf_mat.sum()
            val_loss_avg = loss_sigma / len(valid_loader)
            val_iter_index.append(iteration_number)
            val_loss_list.append(val_loss_avg)
            val_acc_list.append(val_acc_avg)
            print(
                "Validating: Epoch[{:0>3}/{:0>3}] Loss_Avg_Epoch: {:.4f} Accuracy:{:.4f}".format(
                    epoch + 1, opt.max_epoch, val_loss_avg, val_acc_avg))
            # print(euclidean_distance_list)
            # show_distance(euclidean_distance_list)

    print('Finished Training')

    # ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------
    show_plot(train_iter_index, train_loss_list, val_iter_index, val_loss_list, val_acc_list)
    show_distance(euclidean_distance_list)
    # plt.figure(3)
    # show_pairs(net, valid_loader)

    time_str = time.strftime('%m%d%H%M')
    save_path = 'checkpoints/SiameseNet_%s_%s_net_params.pkl' % (time_str, val_acc_avg)
    net_save_path = os.path.join(opt.log_dir, save_path)
    torch.save(net.state_dict(), net_save_path)

    pass


if __name__ == '__main__':
    train()
