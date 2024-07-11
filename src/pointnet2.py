import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import PointNetSetAbstraction,PointNetFeaturePropagation


class pointnet2(nn.Module):
    def __init__(self, num_classes):
        super(pointnet2, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 0 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz[:, 3:, :]
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points

# f = open("data_6.txt", "r")
# data = np.zeros([2, 1000, 3])
# k = 0
# for line in f.readlines():
#     line = line[:-1]
#     a, b, c = line.split()
#     data[0][k][0] = data[1][k][0] = float(a)
#     data[0][k][1] = data[1][k][1] = float(b)
#     data[0][k][2] = data[1][k][2] = float(c)
#
# aa = pointnet2(128)
# data = torch.from_numpy(data).transpose(1, 2).float()
# print(data.shape)
# k = aa(data)
# print(k[0].shape)