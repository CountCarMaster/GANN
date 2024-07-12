import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def knn(x, k):
    """
    get k nearest neighbors.
    :param x: input torch tensors, size (B, N, C)
    :param k: neighbor size.
    :return:
    """
    xsqu = torch.sum(x ** 2, dim=1, keepdim=True)
    xx = torch.matmul(x.transpose(2, 1), x)
    distance = -xsqu - xsqu.transpose(2, 1) + 2 * xx
    idx = distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature1(x, k):
    """
    aggressive the neighbor feature
    :param x: input torch tensors, size (B, N, C)
    :param k: k nearest neighbors.
    :return:
    """
    device = x.device
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k).to(device)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous().to(device)
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
        device = x.device
        x_expand = torch.unsqueeze(x, -2).to(device)
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
        feature = self.mlps(feature.float()).permute(0, 2, 3, 1)
        feature = torch.max(feature, 2)[0]
        feature = torch.cat([x.to(device), feature], dim=-1)
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
    def __init__(self, num_points, in_channel, k_tmp=50, k=20):
        super(new_knn, self).__init__()
        self.k_tmp = k_tmp
        self.k = k
        self.N = num_points
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
        # print(ans.permute(0, 3, 2, 1).shape)
        return ans.permute(0, 3, 2, 1)  # [B, C, k, N]

# pointnet2
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

# DGCNN
def keep_feature(x, k):
    B, C, N = x.shape
    idx = knn(x, k).transpose(1, 2)
    x_expand = x.unsqueeze(2).expand(B, C, k, N)
    idx_expand = idx.unsqueeze(1).expand(B, C, k, N)
    neighbors = torch.gather(x_expand, 3, idx_expand)
    neighbors += x_expand
    return neighbors