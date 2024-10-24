import argparse
import yaml
from src.pointnet import pointnet_cls, pointnet_seg
from src.pointnet2 import pointnet2_cls, pointnet2_seg
from src.DGCNN_with_kmeans import DGCNN_cls, DGCNN_seg
from src.ErrorClass import KeyError
from src.DatasetMaker import ModelNet40, ShapeNet, S3DISFinal
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.metrics import confusion_matrix
import open3d as o3d

def visualizer(data: np.array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])

def datasetChooser(mission: str, batchSize: int, mode: str):
    if mission == 'classification':
        dataset = ModelNet40(batch_size=batchSize, mode=mode)
        return dataset.Load()
    elif mission == 'part segmentation':
        dataset = ShapeNet(batch_size=batchSize, mode=mode)
        return dataset.Load()
    elif mission == 'scene segmentation':
        dataset = S3DISFinal(batch_size=batchSize, mode=mode)
        return dataset.Load()
    else:
        raise KeyError('Unknown mission name. Please choose the mission in '
                       'classification, part segmentation, scene segmentation.')


def modelChooser(modelName: str, mission: str, device: torch.device, k=40, kmax=20):
    if modelName == 'PointNet':
        if mission == 'classification':
            return pointnet_cls(input_channel=3, output_channel=40).to(device)
        elif mission == 'part segmentation':
            return pointnet_seg(input_channel=3, output_channel=6).to(device)
        elif mission == 'scene segmentation':
            return pointnet_seg(input_channel=9, output_channel=13).to(device)
        else:
            raise KeyError('Unknown mission name. Please choose the mission in '
                           'classification, part segmentation, scene segmentation.')
    elif modelName == 'PointNet++':
        if mission == 'classification':
            return pointnet2_cls(num_classes=40).to(device)
        elif mission == 'part segmentation':
            return pointnet2_seg(input_channel=3, num_classes=6).to(device)
        elif mission == 'scene segmentation':
            return pointnet2_seg(input_channel=9, num_classes=13).to(device)
        else:
            raise KeyError('Unknown mission name. Please choose the mission in '
                           'classification, part segmentation, scene segmentation.')
    elif modelName == 'DGCNN':
        if mission == 'classification':
            return DGCNN_cls(input_channel=3, output_channel=40, k=k).to(device)
        elif mission == 'part segmentation':
            return DGCNN_seg(input_channel=3, output_channel=6, k=k).to(device)
        elif mission == 'scene segmentation':
            return DGCNN_seg(input_channel=9, output_channel=13, k=k).to(device)
        else:
            raise KeyError('Unknown mission name. Please choose the mission in '
                           'classification, part segmentation, scene segmentation.')
    # elif modelName == 'GANN':
    #     if mission == 'classification':
    #         return GANN_cls(input_channel=3, output_channel=40, k=k, kmax=kmax).to(device)
    #     elif mission == 'part segmentation':
    #         return GANN_seg(input_channel=3, output_channel=6, k=k, kmax=kmax).to(device)
    #     elif mission == 'scene segmentation':
    #         return GANN_seg(input_channel=9, output_channel=13, k=k, kmax=kmax).to(device)
    #     else:
    #         raise KeyError('Unknown mission name. Please choose the mission in '
    #                        'classification, part segmentation, scene segmentation.')
    else:
        raise KeyError('Unknown model name. Please choose the model in '
                       'pointnet, pointnet2, DGCNN and GANN')

def optimizerChooser(model, optimizerName: str, learningRate: float):
    if optimizerName == 'Adam':
        return optim.Adam(model.parameters(), lr=learningRate)
    elif optimizerName == 'SGD':
        return optim.SGD(model.parameters(), lr=learningRate)
    elif optimizerName == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=learningRate)
    elif optimizerName == 'Nadam':
        return optim.NAdam(model.parameters(), lr=learningRate)
    else:
        raise KeyError('Unknown optimizer name. Please choose the optimizer in '
                       'Adam, SGD, RMSprop and Nadam')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--yaml-path', type=str, default='./config.yaml')
    args = parser.parse_args()

    with open(args.yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    device = torch.device(config['device'])
    dataLoader = datasetChooser(config['mission'], config['batchSize'], config['mode'])  # [B, N, C]
    model = modelChooser(config['model'], config['mission'], device, config['k'], config['kmax'])


    testDataLoader = None
    if config['val'] == 1:
        testDataLoader = datasetChooser(config['mission'], config['datasetRootDir'], config['batchSize'], 'test')

    if config['mode'] == 'train':
        # writer = SummaryWriter(config['summaryLogDir'])
        if config['loadModelDir'] != '0':
            model.load_state_dict(torch.load(config['loadModelDir']))
        optimizer = optimizerChooser(model, config['optimizer'], config['learningRate'])
        criterion = nn.CrossEntropyLoss().to(device)
        model.train()
        loss = None
        minLoss = 1000000
        maxAcc = -1
        lossArray = []
        for epoch in range(config['epochs']):
            for x, y in dataLoader:
                x = x.to(device).transpose(1, 2)
                x = x.float()
                y = y.to(device)
                output = model(x)
                loss = criterion(output.type(torch.float32), y.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # writer.add_scalar('Loss', loss.item(), epoch)
            lossArray.append(loss.item())

            allRes = 0
            allAcc = 0
            acc = 0
            with torch.no_grad():
                for x, y in testDataLoader:
                    x = x.to(device).transpose(1, 2)
                    x = x.float()
                    y = y.to(device)
                    output = model(x)
                    result = torch.max(output, dim=1)[1]
                    allRes += result.shape[0]
                    allAcc += torch.sum(result == y.long())
                acc = allAcc * 1.0 / allRes * 1.0

            # if loss.item() < minLoss:
            #     minLoss = loss.item()
            #     torch.save(model.state_dict(), '%s/%s_best_loss.pt' % (config['modelSaveDir'], config['model']))

            print(f'Epoch [{epoch + 1}/{config["epochs"]}], Loss: {loss.item():.4f}')