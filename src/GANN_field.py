import torch
import torch.nn as nn
from src.utils import keep_feature1

class DGCNN_cls(nn.Module):
    def __init__(self, input_channel, output_channel, k=20):
        super(DGCNN_cls, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(128)

        self.edge_cov1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.ReLU()
        )
        self.edge_cov2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.ReLU()
        )
        self.edge_cov3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.ReLU()
        )
        self.edge_cov4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.ReLU()
        )
        self.edge_cov5 = nn.Sequential(
            nn.Conv1d(64, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.ReLU()
        )
        self.edge_cov6 = nn.Sequential(
            nn.Linear(1024, 512),
            self.bn6,
            nn.ReLU()
        )
        self.edge_cov7 = nn.Sequential(
            nn.Linear(512, 128),
            self.bn7,
            nn.ReLU()
        )
        self.edge_cov8 = nn.Sequential(
            nn.Linear(128, output_channel),
            nn.ReLU()
        )

    def forward(self, x): #[B, C, N]
        B, C, N = x.shape
        x = keep_feature1(x, self.k) # [B, C, k, N]
        x = self.edge_cov1(x)
        x = torch.max(x, dim=2)[0]

        x = keep_feature1(x, self.k)
        x = self.edge_cov2(x)
        x1 = torch.max(x, dim=2)[0]

        x = keep_feature1(x1, self.k)
        x = self.edge_cov3(x)
        x = torch.max(x, dim=2)[0]

        x = keep_feature1(x, self.k)
        x = self.edge_cov4(x)
        x = torch.max(x, dim=2)[0]

        x = self.edge_cov5(x)
        x = torch.max(x, dim=2)[0]

        x = self.edge_cov6(x)
        x = self.edge_cov7(x)
        x = self.edge_cov8(x)
        return x