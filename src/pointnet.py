import numpy as np
import torch
import torch.nn as nn

# device = torch.device('cuda')

class pointnet(nn.Module):
    def __init__(self, output_channel):
        super(pointnet, self).__init__()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)

        self.edge_cov1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.ReLU()
        )
        self.edge_cov2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.ReLU()
        )
        self.edge_cov3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.ReLU()
        )
        self.edge_cov4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            self.bn4,
            nn.ReLU()
        )
        self.edge_cov5 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.ReLU()
        )
        self.edge_cov6 = nn.Sequential(
            nn.Conv1d(1088, 512, kernel_size=1, bias=False),
            self.bn6,
            nn.ReLU()
        )
        self.edge_cov7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            self.bn7,
            nn.ReLU()
        )
        self.edge_cov8 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            self.bn8,
            nn.ReLU()
        )
        self.edge_cov9 = nn.Sequential(
            nn.Conv1d(128, output_channel, kernel_size=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x): #[B, C, N]
        B, C, N = x.shape
        x = self.edge_cov1(x)
        x1 = self.edge_cov2(x)
        x = self.edge_cov3(x1)
        x = self.edge_cov4(x)
        x = self.edge_cov5(x)
        # [B, 1024, N]
        globalFeature = torch.max(x, dim=2)[0]
        # [B, 1024]
        globalFeature = globalFeature.view(B, 1024, 1).repeat(1, 1, N)
        x = torch.cat((x1, globalFeature), dim=1)
        # [B, 1088, N]
        x = self.edge_cov6(x)
        x = self.edge_cov7(x)
        x = self.edge_cov8(x)
        x = self.edge_cov9(x)
        return x

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
# aa = pointnet(128)
# k = aa(torch.from_numpy(data).transpose(1, 2).float())
# print(k.shape)