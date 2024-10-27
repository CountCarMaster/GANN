import torch
import torch.nn as nn
from src.utils import keep_feature3, get_graph_feature1, xyz2sphere, resort_points

class GeometricExtractor(nn.Module):
    def __init__(self):
        super(GeometricExtractor, self).__init__()
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(10)
        self.mlp1 = nn.Sequential(
            nn.Linear(10, 10),
            self.bn1,
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(10, 10),
            self.bn2,
            nn.ReLU()
        )

    def forward(self, x, k=9):
        B, N, C = x.shape
        device = x.device
        x_expand = torch.unsqueeze(x, -2).to(device)
        x_neighbour = get_graph_feature1(x.transpose(1, 2), k + 1) - x_expand
        x_neighbour = x_neighbour[:, :, 1:, :]
        x_sphere = xyz2sphere(x_neighbour)
        phi = x_sphere[..., 2]
        idx = phi.argsort(dim=-1)
        pairs = resort_points(x_neighbour, idx).unsqueeze(-2)
        pairs = torch.cat((pairs, torch.roll(pairs, -1, dims=-3)), dim=-2)
        centroids = torch.mean(pairs, dim=-2)
        vector_1 = pairs[..., 0, :].view(B, N, k, -1)
        vector_2 = pairs[..., 1, :].view(B, N, k, -1)
        normals = torch.cross(vector_1, vector_2, dim=-1)  # [B, N, k, 3]
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)
        pos_mask = (normals[..., 0:1, 0] > 0).float() * 2. - 1.
        normals *= torch.unsqueeze(pos_mask, -1)
        position = torch.sum(normals * centroids, dim=-1) / torch.sqrt(torch.tensor(3))
        position = position.unsqueeze(-1)

        dot_product = torch.sum(vector_1 * vector_2, dim=-1)
        norm_A = torch.norm(vector_1, dim=-1)
        norm_B = torch.norm(vector_2, dim=-1)
        cos_theta = dot_product / (norm_A * norm_B + 1e-8)
        angles = torch.acos(torch.clamp(cos_theta, -1.0, 1.0)).unsqueeze(-1)

        norm_A = norm_A.unsqueeze(-1)
        norm_B = norm_B.unsqueeze(-1)
        feature = torch.cat((centroids, normals, position, angles, norm_A, norm_B), dim=-1)
        feature = feature.view(B * N * k, -1)

        feature = self.mlp1(feature)
        feature = self.mlp2(feature)
        feature = feature.view(B, N, k, -1)
        feature = torch.max(feature, dim=-2)[0]
        return feature

class GANN_cls(nn.Module):
    def __init__(self, input_channel, output_channel, k=20):
        super(GANN_cls, self).__init__()
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

        self.geometric_extractor = GeometricExtractor()

        self.edge_cov1 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=1, bias=False),
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
        x = keep_feature3(x, self.k) # [B, C, k, N]
        x = self.edge_cov1(x)
        x = torch.max(x, dim=2)[0]

        x = keep_feature3(x, self.k)
        x = self.edge_cov2(x)
        x1 = torch.max(x, dim=2)[0]

        x = keep_feature3(x1, self.k)
        x = self.edge_cov3(x)
        x = torch.max(x, dim=2)[0]

        x = keep_feature3(x, self.k)
        x = self.edge_cov4(x)
        x = torch.max(x, dim=2)[0]

        x = self.edge_cov5(x)
        x = torch.max(x, dim=2)[0]

        x = self.edge_cov6(x)
        x = self.edge_cov7(x)
        x = self.edge_cov8(x)
        return x
