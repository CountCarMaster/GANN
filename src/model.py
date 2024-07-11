import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda')

def knn(x, k):
    xsqu = torch.sum(x ** 2, dim=1, keepdim=True)
    xx = torch.matmul(x.transpose(2, 1), x)
    distance = -xsqu - xsqu.transpose(2, 1) + 2 * xx
    idx = distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k):
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def get_graph_feature1(x, k):
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    return feature


def xyz2sphere(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (rho, theta, phi) [B, N, 3] / [B, N, G, 3]
    """
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, min=0)  # range: [0, inf]
    theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = torch.cat([rho, theta, phi], dim=-1)
    return out


def resort_points(points, idx):
    """
    Resort Set of points along G dim

    """
    device = points.device
    B, N, G, _ = points.shape

    view_shape = [B, 1, 1]
    repeat_shape = [1, N, G]
    b_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    view_shape = [1, N, 1]
    repeat_shape = [B, 1, G]
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[b_indices, n_indices, idx, :]

    return new_points


class umbrella_repsurf(nn.Module):
    def __init__(self, k=9, random_inv=False, in_channel=7):
        super(umbrella_repsurf, self).__init__()
        self.k = k
        self.random_inv = random_inv
        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
        )

    def forward(self, x):
        B, N, C = x.shape
        x_expand = torch.unsqueeze(x, -2)
        x_neighbour = get_graph_feature1(x.transpose(1, 2), self.k + 1) - x_expand
        x_neighbour = x_neighbour[:, :, 1:, :]
        x_sphere = xyz2sphere(x_neighbour)
        phi = x_sphere[..., 2]
        idx = phi.argsort(dim=-1)
        pairs = resort_points(x_neighbour, idx).unsqueeze(-2)
        pairs = torch.cat((pairs, torch.roll(pairs, -1, dims=-3)), dim=-2)
        centroids = torch.mean(pairs, dim=-2)
        vector_1 = pairs[..., 0, :].view(B, N, self.k, -1)
        vector_2 = pairs[..., 1, :].view(B, N, self.k, -1)
        normals = torch.cross(vector_1, vector_2, dim=-1) + torch.tensor(0.00001)  # [B, N, k, 3]
        normals = normals / torch.norm(normals, dim=-1, keepdim=True)
        pos_mask = (normals[..., 0:1, 0] > 0).float() * 2. - 1.
        normals *= torch.unsqueeze(pos_mask, -1)
        if self.random_inv:
            random_mask = torch.randint(0, 2, (x.size(0), 1, 1)).float() * 2. - 1.
            random_mask = random_mask.to(normals.device)
            normals = normals * random_mask.unsqueeze(-1)
        positions = torch.sum(centroids * normals, dim=3) / torch.sqrt(torch.tensor(3).to(device))
        feature = torch.cat((centroids, normals, positions.unsqueeze(-1)), dim=-1)
        feature = torch.permute(feature, (0, 3, 1, 2))
        feature = self.mlps(feature).permute(0, 2, 3, 1)
        feature = torch.max(feature, 2)[0]
        feature = torch.cat([x, feature], dim=-1)
        return feature


class AddNorm(nn.Module):
    def __init__(self, dim):
        super(AddNorm, self).__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input, x):
        input = self.dropout(input)
        return self.norm(input + x)
class new_knn(nn.Module):
    def __init__(self, in_channel, k_tmp=50, k=20):
        super(new_knn, self).__init__()
        self.k_tmp = k_tmp
        self.k = k
        self.N = 1000
        self.qLinear = nn.Linear(self.N, self.N)
        self.kLinear = nn.Linear(self.N, self.N)
        self.vLinear = nn.Linear(self.N, self.N)
        self.fc = nn.Linear(in_channel, 1)
        self.dense = nn.Linear(self.N, self.N)
        self.in_channel = in_channel
        self.addnorm1 = AddNorm(self.N)
        self.addnorm2 = AddNorm(self.N)

    def forward(self, x):
        B, C, N = x.shape  # [B, C, N]
        self.N = N
        x_with_neighbour = get_graph_feature1(x, self.k_tmp)
        x_copied = x_with_neighbour
        x_unsqueezed = x.transpose(1, 2).unsqueeze(-2)
        x_with_neighbour -= x_unsqueezed
        x_with_neighbour = x_with_neighbour.permute(0, 2, 3, 1)  # [B, k, C, N]
        q = self.qLinear(x_with_neighbour.view(B, -1, N).contiguous())
        k = self.kLinear(x_with_neighbour.view(B, -1, N).contiguous())
        v = self.vLinear(x_with_neighbour.view(B, -1, N).contiguous())
        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1).contiguous())
        scaled_attention_logits /= torch.sqrt(torch.tensor(self.N, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        output = self.addnorm1(output, x_with_neighbour.view(B, -1, N))
        x = self.dense(output)
        x = self.addnorm2(x, output)
        x = x.view(B, -1, C, self.N).contiguous()
        x += x_with_neighbour
        x = x.permute(0, 3, 1, 2)
        x = self.fc(x)  # [B, N, k_tmp, 1]
        x = x.view(B, N, -1).contiguous()  # [B, N, k_tmp]
        _, indices = torch.sort(x)
        indices = indices[:, :, :self.k]  # [B, N, k]
        ans = torch.gather(x_copied, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, C))
        return ans.permute(0, 3, 2, 1)  # [B, C, k, N]

class DGCNN(nn.Module):
    def __init__(self, k, output_channel):
        super(DGCNN, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.edge_cov1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.ReLU()
        )
        self.edge_cov2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.ReLU()
        )
        self.edge_cov3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.ReLU()
        )
        self.edge_cov4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.ReLU()
        )
        self.edge_cov5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.ReLU()
        )

        self.mlp1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.mlp2 = nn.Linear(512, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.mlp3 = nn.Linear(256, output_channel, bias=False)
        # self.umbrella_repsurf = unbrella_repsurf()

        self.knn_1 = new_knn(in_channel=20)

    def forward(self, x):  # (batch_size, 3, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)

        # x = self.umbrella_repsurf(x)
        x = x.transpose(1, 2)
        idx = x[:, 10000:, 0].type(torch.int64)
        x = x[:, :10000, :]
        x = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        x = x.transpose(1, 2)  # [B, C, N]

        # x = get_graph_feature(x, self.k)  # (batch_size, 6, num_points, k)
        x_neighbour = self.knn_1(x)  # [B, C, k, N]
        x = x.view(batch_size, num_points, 1, 20).repeat(1, 1, 20, 1)
        x = self.edge_cov1(x)  # (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points)

        x = get_graph_feature(x1, self.k)  # (batch_size, 128, num_points, k)
        x = self.edge_cov2(x)  # (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points)

        x = get_graph_feature(x2, self.k)  # (batch_size, 128, num_points, k)
        x = self.edge_cov3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, self.k)  # (batch_size, 256, num_points, k)
        x = self.edge_cov4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)
        x = self.edge_cov5(x)  # (batch_size, 1024, num_points)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (batch_size, 1024)
        x = x.max(dim=-1, keepdim=False)[0]

        x = F.relu(self.bn6(self.mlp1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.mlp2(x)))
        x = self.dp2(x)
        x = self.mlp3(x)

        return x


class DGCNN_repsurf(nn.Module):
    def __init__(self, k, output_channel):
        super(DGCNN_repsurf, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.edge_cov1 = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.ReLU()
        )
        self.edge_cov2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.ReLU()
        )
        self.edge_cov3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.ReLU()
        )
        self.edge_cov4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.ReLU()
        )
        self.edge_cov5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            self.bn5,
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

        self.knn1 = new_knn(in_channel=10)
        self.knn2 = new_knn(in_channel=64)
        self.knn3 = new_knn(in_channel=64)
        self.knn4 = new_knn(in_channel=128)

    def forward(self, x):  # (batch_size, 3, num_points)
        batch_size = x.size(0)

        # x = self.umbrella_repsurf(x)
        x = x.transpose(1, 2)
        idx = x[:, 10000:, 0].type(torch.int64)
        x = x[:, :10000, :]
        # x = self.umbrella_repsurf(x)
        x = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        x = self.umbrella_repsurf(x)
        x = x.transpose(1, 2)

        # x = get_graph_feature(x, self.k)  # (batch_size, 6, num_points, k)
        x_neighbour = self.knn1(x)  # [B, C, k, N]
        x = x.view(batch_size, 10, 1, 1000).repeat(1, 1, 20, 1)
        x = torch.cat((x_neighbour - x, x), dim=1).permute(0, 1, 3, 2)
        x = self.edge_cov1(x)  # (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points)

        # x = get_graph_feature(x1, self.k)  # (batch_size, 128, num_points, k)
        x_neighbour = self.knn2(x1)  # [B, C, k, N]
        x = x1.view(batch_size, 64, 1, 1000).repeat(1, 1, 20, 1)
        x = torch.cat((x_neighbour - x, x), dim=1).permute(0, 1, 3, 2)
        x = self.edge_cov2(x)  # (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points)

        # x = get_graph_feature(x2, self.k)  # (batch_size, 128, num_points, k)
        x_neighbour = self.knn3(x2)  # [B, C, k, N]
        x = x2.view(batch_size, 64, 1, 1000).repeat(1, 1, 20, 1)
        x = torch.cat((x_neighbour - x, x), dim=1).permute(0, 1, 3, 2)
        x = self.edge_cov3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        # x = get_graph_feature(x3, self.k)  # (batch_size, 256, num_points, k)
        x_neighbour = self.knn4(x3)  # [B, C, k, N]
        x = x3.view(batch_size, 128, 1, 1000).repeat(1, 1, 20, 1)
        x = torch.cat((x_neighbour - x, x), dim=1).permute(0, 1, 3, 2)
        x = self.edge_cov4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)
        x = self.edge_cov5(x)  # (batch_size, 1024, num_points)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (batch_size, 1024)
        x = x.max(dim=-1, keepdim=False)[0]

        x = F.relu(self.bn6(self.mlp1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.mlp2(x)))
        x = self.dp2(x)
        x = self.mlp3(x)

        return x

