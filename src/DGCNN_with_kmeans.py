import torch
import torch.nn as nn
from src.utils import keep_feature
from torch_kmeans import KMeans

class DGCNN_cls(nn.Module):
    def __init__(self, input_channel, output_channel, k=15, kmax=40):
        super(DGCNN_cls, self).__init__()
        self.k = k
        self.kmax = kmax
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(128)

        self.KMeans = KMeans(cluser=k, device=torch.device('cuda:0'), verbose=False)

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

    def neighbor_finder(self, x: torch.Tensor):
        # [B, C, k, N]
        B, C, k, N = x.shape
        device = x.device
        x = x.view(-1, k, C)
        cluster_result = self.KMeans(x)
        ans = torch.zeros(B, C, self.k, N).to(device)
        ans = ans.view(self.k, -1, C)
        for i in range(self.k):
            mask = (cluster_result.labels == i).to(device)
            sum_tensor = torch.zeros(B * N, C).to(device)
            count_tensor = torch.zeros(B * N).to(device)
            for j in range(self.k):
                sum_tensor += (x[:, j] * mask[:, j].unsqueeze(1))
                count_tensor += mask[:, j].float()
            count_tensor[count_tensor == 0] = 1
            average_result = sum_tensor / count_tensor.unsqueeze(1)
            ans[i] = average_result
        ans = ans.view(B, C, self.k, N)
        return ans

    def forward(self, x): #[B, C, N]
        B, C, N = x.shape
        x = keep_feature(x, self.kmax) # [B, C, k, N]
        x = self.neighbor_finder(x)
        x = self.edge_cov1(x)
        x = torch.max(x, dim=2)[0]

        x = keep_feature(x, self.kmax)
        x = self.neighbor_finder(x)
        x = self.edge_cov2(x)
        x1 = torch.max(x, dim=2)[0]

        x = keep_feature(x1, self.kmax)
        x = self.neighbor_finder(x)
        x = self.edge_cov3(x)
        x = torch.max(x, dim=2)[0]

        x = keep_feature(x, self.kmax)
        x = self.neighbor_finder(x)
        x = self.edge_cov4(x)
        x = torch.max(x, dim=2)[0]

        x = self.edge_cov5(x)
        x = torch.max(x, dim=2)[0]

        x = self.edge_cov6(x)
        x = self.edge_cov7(x)
        x = self.edge_cov8(x)
        return x

class DGCNN_seg(nn.Module):
    def __init__(self, input_channel, output_channel, k=15, kmax=40):
        super(DGCNN_seg, self).__init__()
        self.k = k
        self.kmax = kmax
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(128)

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
            nn.Conv1d(1088, 512, kernel_size=1, bias=False),
            self.bn6,
            nn.ReLU()
        )
        self.edge_cov7 = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1, bias=False),
            self.bn7,
            nn.ReLU()
        )
        self.edge_cov8 = nn.Sequential(
            nn.Conv1d(128, output_channel, kernel_size=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x): #[B, C, N]
        B, C, N = x.shape
        x = keep_feature(x, self.kmax) # [B, C, k, N]
        x = self.edge_cov1(x)
        x = torch.max(x, dim=2)[0]

        x = keep_feature(x, self.kmax)
        x = self.edge_cov2(x)
        x1 = torch.max(x, dim=2)[0]

        x = keep_feature(x1, self.kmax)
        x = self.edge_cov3(x)
        x = torch.max(x, dim=2)[0]

        x = keep_feature(x, self.kmax)
        x = self.edge_cov4(x)
        x = torch.max(x, dim=2)[0]

        x = self.edge_cov5(x)
        x = torch.max(x, dim=2)[0]

        x = torch.cat([x1, x.unsqueeze(-1).repeat(1, 1, N)], dim=1)

        x = self.edge_cov6(x)
        x = self.edge_cov7(x)
        x = self.edge_cov8(x)
        return x

