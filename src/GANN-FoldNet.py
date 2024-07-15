import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import umbrella_repsurf, new_knn, FoldingNet

class GANN(nn.Module):
    def __init__(self, num_points, k, output_channel):
        """
        GANN Model
        :param num_points: the point number of the input tensor.
        :param k: hyperparameter -> number of neighbours.
        :param output_channel: output class number.
        """
        super(GANN, self).__init__()
        self.k = k
        self.num_points = num_points
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.fn1 = FoldingNet()
        self.fn2 = FoldingNet()
        self.fn3 = FoldingNet()
        self.fn4 = FoldingNet()

        self.edge_cov1 = nn.Sequential(
            nn.Conv2d(20+512, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.ReLU()
        )
        self.edge_cov2 = nn.Sequential(
            nn.Conv2d(128+512, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.ReLU()
        )
        self.edge_cov3 = nn.Sequential(
            nn.Conv2d(128+512, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.ReLU()
        )
        self.edge_cov4 = nn.Sequential(
            nn.Conv2d(256+512, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.ReLU()
        )
        self.edge_cov5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.ReLU()
        )

        self.edge_cov6 = nn.Sequential(
            nn.Conv1d(1152, 512, kernel_size=1, bias=False),
            self.bn6,
            nn.ReLU()
        )

        self.edge_cov7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            self.bn7,
            nn.ReLU()
        )

        self.edge_cov8 = nn.Sequential(
            nn.Conv1d(256, output_channel, kernel_size=1, bias=False),
            nn.ReLU()
        )

        self.mlp1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.mlp2 = nn.Linear(512, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.mlp3 = nn.Linear(256, output_channel, bias=False)
        self.umbrella_repsurf = umbrella_repsurf()

        self.knn1 = new_knn(in_channel=10, num_points=self.num_points, k=self.k)
        self.knn2 = new_knn(in_channel=64, num_points=self.num_points, k=self.k)
        self.knn3 = new_knn(in_channel=64, num_points=self.num_points, k=self.k)
        self.knn4 = new_knn(in_channel=128, num_points=self.num_points, k=self.k)

        self.mlp1_1 = nn.Conv2d(10, 10, kernel_size=1, bias=False)
        self.mlp1_2 = nn.Conv2d(10, 10, kernel_size=1, bias=False)
        self.softmax1 = nn.Softmax(dim=1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.mlp2_1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.mlp2_2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.softmax2 = nn.Softmax(dim=1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.mlp3_1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.mlp3_2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.softmax3 = nn.Softmax(dim=1)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.mlp4_1 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.mlp4_2 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.softmax4 = nn.Softmax(dim=1)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):  # (batch_size, 3, num_points)
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        # [B, N, C]
        x = self.umbrella_repsurf(x)
        x = x.transpose(1, 2)
        # [B, C, N=10]

        x_neighbour = self.knn1(x)  # [B, C, k, N]
        globalFeature = self.fn1(x, x_neighbour).unsqueeze(-1).repeat(1, 1, 1, self.k)  # [B, C, N, k]
        x = x.view(batch_size, 10, 1, self.num_points).repeat(1, 1, self.k, 1)
        alpha = self.softmax1(self.leaky_relu1(self.mlp1_1(x_neighbour - x) + self.mlp1_2(x)))
        x = torch.cat((alpha*(x_neighbour - x), x), dim=1).permute(0, 1, 3, 2)
        x = torch.cat((x, globalFeature), dim=1)
        # [B, C*2+512, N, k]
        x = self.edge_cov1(x)
        # [B, 64, N, k]
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points)

        x_neighbour = self.knn2(x1)  # [B, C, k, N]
        globalFeature = self.fn2(x1, x_neighbour).unsqueeze(-1).repeat(1, 1, 1, self.k)
        x = x1.view(batch_size, 64, 1, self.num_points).repeat(1, 1, self.k, 1)
        alpha = self.softmax2(self.leaky_relu2(self.mlp2_1(x_neighbour - x) + self.mlp2_2(x)))
        x = torch.cat((alpha * (x_neighbour - x), x), dim=1).permute(0, 1, 3, 2)
        x = torch.cat((x, globalFeature), dim=1)
        x = self.edge_cov2(x)  # (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points)

        x_neighbour = self.knn3(x2)  # [B, C, k, N]
        globalFeature = self.fn3(x2, x_neighbour).unsqueeze(-1).repeat(1, 1, 1, self.k)
        x = x2.view(batch_size, 64, 1, self.num_points).repeat(1, 1, self.k, 1)
        alpha = self.softmax3(self.leaky_relu3(self.mlp3_1(x_neighbour - x) + self.mlp3_2(x)))
        x = torch.cat((alpha * (x_neighbour - x), x), dim=1).permute(0, 1, 3, 2)
        x = torch.cat((x, globalFeature), dim=1)
        x = self.edge_cov3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x_neighbour = self.knn4(x3)  # [B, C, k, N]
        globalFeature = self.fn4(x3, x_neighbour).unsqueeze(-1).repeat(1, 1, 1, self.k)
        x = x3.view(batch_size, 128, 1, self.num_points).repeat(1, 1, self.k, 1)
        alpha = self.softmax4(self.leaky_relu4(self.mlp4_1(x_neighbour - x) + self.mlp4_2(x)))
        x = torch.cat((alpha * (x_neighbour - x), x), dim=1).permute(0, 1, 3, 2)
        x = torch.cat((x, globalFeature), dim=1)
        x = self.edge_cov4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)
        x = self.edge_cov5(x)  # (batch_size, 1024, num_points)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (batch_size, 1024)
        x = x.max(dim=-1, keepdim=False)[0].view(batch_size, -1, 1)
        x = x.repeat(1, 1, self.num_points)
        x = torch.cat((x3, x), dim=1)

        x = self.edge_cov6(x)
        x = self.edge_cov7(x)
        x = self.edge_cov8(x)

        return x


device = torch.device('cuda')
model = GANN(1000, 5, 128).to(device)
data = torch.rand(20, 3, 1000).to(device)