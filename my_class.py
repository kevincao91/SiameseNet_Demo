from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os


class LWFDataset(Dataset):
    def __init__(self, data_dir, split='pairsDevTrain', transform=None, target_transform=None):
        img_dir = os.path.join(data_dir, 'lfw/')
        txt_path = os.path.join(data_dir, '{0}.txt'.format(split))
        fh = open(txt_path, 'r')
        imgs0 = []
        imgs1 = []
        label = []
        pos_num = 0
        neg_num = 0
        line_index = 1
        for line in fh:
            # print(line_index)
            line = line.rstrip()
            words = line.split()
            if line_index == 1:
                pos_num = int(words[0])
                neg_num = pos_num
            elif line_index <= pos_num+1:  # positive pairs
                img_file_path0 = os.path.join(img_dir, words[0], words[0] + "_%0*d.jpg" % (4, int(words[1])))
                # print(img_file_path0)
                imgs0.append(img_file_path0)
                img_file_path1 = os.path.join(img_dir, words[0], words[0] + "_%0*d.jpg" % (4, int(words[2])))
                # print(img_file_path1)
                imgs1.append(img_file_path1)
                value = 0
                label.append(value)
            else:  # negative pairs
                img_file_path0 = os.path.join(img_dir, words[0], words[0] + "_%0*d.jpg" % (4, int(words[1])))
                # print(img_file_path0)
                imgs0.append(img_file_path0)
                img_file_path1 = os.path.join(img_dir, words[2], words[2] + "_%0*d.jpg" % (4, int(words[3])))
                # print(img_file_path1)
                imgs1.append(img_file_path1)
                value = 1
                label.append(value)
            line_index += 1

        self.imgs0 = imgs0  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgs1 = imgs1  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn0 = self.imgs0[index]
        fn1 = self.imgs1[index]
        img0 = Image.open(fn0).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img1 = Image.open(fn1).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        label = self.label[index]

        if self.transform is not None:
            img0 = self.transform(img0)  # 在这里做transform，转为tensor等等
            img1 = self.transform(img1)  # 在这里做transform，转为tensor等等

        return img0, img1, label

    def __len__(self):
        return len(self.imgs0)


class SiameseNetworkSimple(nn.Module):
    def __init__(self):
        super(SiameseNetworkSimple, self).__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 5, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(5),
            nn.Dropout2d(p=0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(5, 7, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(7),
            nn.Dropout2d(p=0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 100)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseNetworkWED(nn.Module):
    def __init__(self):
        super(SiameseNetworkWED, self).__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 5, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(5),
        )

        self.fc = nn.Sequential(
            nn.Linear(5 * 100 * 100, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(100, 100)
        )

        self.classifier = nn.Sequential(
            nn.Linear(100, 1),
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = self.classifier(torch.abs(output2-output1))
        return output


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 6, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(6),
            nn.Dropout2d(p=0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(6, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(12, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(12 * 200 * 200, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 12)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                        2))

        return loss_contrastive
