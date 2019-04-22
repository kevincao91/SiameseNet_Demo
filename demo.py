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

from config import opt
from my_class import LWFDataset, SiameseNetwork, ContrastiveLoss, SiameseNetworkSimple


# Helper functions
def show_pairs(test_data_num, inputs0, inputs1, euclidean_distance_list):
    for i in range(test_data_num):
        plt.figure(i)
        img0 = inputs0[i].transpose(1, 2, 0)
        img1 = inputs1[i].transpose(1, 2, 0)
        img2 = plt.imread('/home/kevin/文档/LFW/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')
        two_img = np.hstack((img0, img1))
        plt.imshow(two_img)
        plt.axis("off")
        text = format(euclidean_distance_list[i], '0.2f')
        w = np.shape(img0)[0]
        plt.text(w - 5, 30, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        plt.show()


if __name__ == '__main__':
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
        transforms.Resize(100),
        transforms.ToTensor(),
        normTransform
    ])
    # print('load data')

    # 构建MyDataset实例
    test_data = LWFDataset(data_dir=opt.lwf_data_dir, split='pairsDemoTest', transform=testTransform)
    test_data_num = len(test_data)
    print(test_data_num)

    # 构建DataLoder
    test_loader = DataLoader(dataset=test_data, batch_size=test_data_num)

    print('load data done !')

    # ------------------------------------ step 2/5 : 定义网络------------------------------------
    # print('load net ')
    net = SiameseNetworkSimple()  # 创建一个网络
    net_info = torch.load('/home/kevin/PycharmProjects/SiameseNetwork/checkpoint/'
                          '0213-1633/0213-1633-0.52-0.95-net_params.pkl')
    net.load_state_dict(net_info)
    print('load net done !')

    it = iter(test_loader)
    inputs0, inputs1, labels = it.next()
    inputs0, inputs1, labels = Variable(inputs0), Variable(inputs1), Variable(labels)
    outputs0, outputs1 = net(inputs0, inputs1)
    # 统计
    euclidean_distance = F.pairwise_distance(outputs0, outputs1)
    euclidean_distance_list = euclidean_distance.data.numpy()
    # 预处理
    inputs0, inputs1 = inputs0.data.numpy(), inputs1.data.numpy()

    show_pairs(test_data_num, inputs0, inputs1, euclidean_distance_list)
