import argparse
import yaml
from src.pointnet import pointnet
from src.pointnet2 import pointnet2
from src.DGCNN import DGCNN
from src.GDANet import GDANet
from src.GANN import GANN
from src.ErrorClass import KeyError
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os

def dataloader(config: dict, device) -> DataLoader:
    fileNum = config['inputEndNum'] - config['inputBeginNum'] + 1
    dataX = torch.zeros((fileNum, config['pointNum'], config['inputFeatureChannelNum'])).float()
    dataY = torch.zeros((fileNum, config['pointNum'])).int()
    for i in range(config['inputBeginNum'], config['inputEndNum'] + 1):
        num = i - config['inputBeginNum']
        xPath = os.path.join(config['inputDataDir'], config['inputDataFileName'] + str(i) + '.txt')
        yPath = os.path.join(config['inputLabelDir'], config['inputLabelFileName'] + str(i) + '.txt')
        xFile = open(xPath, 'r')
        yFile = open(yPath, 'r')
        xlis = list()
        for line in xFile.readlines():
            line = line[: -1]
            xlis.append(line.split())
        xFile.close()
        dataX[num, :, :] = torch.tensor(np.array(xlis, dtype=float))
        k = 0
        for line in yFile.readlines():
            line = line[: -1]
            dataY[num][k] = int(line)
            k += 1
        yFile.close()

    dataX = dataX.transpose(1, 2).to(device)  # [B, C, N]
    dataY = dataY.to(device)  # [B, N]
    dataset = TensorDataset(dataX, dataY)
    dataLoader = DataLoader(dataset, batch_size=config['batchSize'], shuffle=True, drop_last=True)
    return dataLoader

def modelChooser(modelName: str, classNum: int, device, k=5, num_points=1000):
    if modelName == 'pointnet':
        return pointnet(classNum).to(device)
    elif modelName == 'pointnet2':
        return pointnet2(classNum).to(device)
    elif modelName == 'DGCNN':
        return DGCNN(classNum, k).to(device)
    elif modelName == 'GDANet':
        return GDANet(classNum).to(device)
    elif modelName == 'GANN':
        return GANN(num_points, k, classNum).to(device)
    else:
        raise KeyError('Unknown model name. Please choose the model in '
                       'pointnet, pointnet2, DGCNN, GDANet and GANN')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--yaml-path', type=str, default='./config.yaml')
    args = parser.parse_args()

    with open(args.yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    device = torch.device(config['device'])
    dataLoader = dataloader(config, device)
    model = modelChooser(config['model'], config['numClass'], device, config['k'], config['pointNum'])
    optimizer = optim.Adam(model.parameters(), lr=config['learningRate'])
    criterion = nn.CrossEntropyLoss().to(device)

    if config['mode'] == 'train':
        minLoss = 100000
        maxAcc = 0
        model.train()
        writer = SummaryWriter(config['summaryLogDir'])
        for epoch in range(config['epochs']):
            loss = None
            for batch_x, batch_y in dataLoader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x).transpose(2, 1)
                loss = criterion(output.type(torch.float32).view(-1, config['numClass']), batch_y.type(torch.int64).view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            writer.add_scalar('Loss', loss.item(), epoch)
            if(loss.item() < minLoss):
                minLoss = loss.item()
                torch.save(model.state_dict(), '%s/%s_best_loss.pt' % (config['modelSaveDir'], config['model']))


            if config['val'] == 1:
                tot_correct = 0
                tot_points = 0
                with torch.no_grad():
                    for batch_x, batch_y in dataLoader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        output = model(batch_x)

                        _, predicted = torch.max(output, 1)
                        correct = (predicted == batch_y).sum().item()
                        tot_correct += correct
                        tot_points += batch_y.numel()
                acc = tot_correct / tot_points
                writer.add_scalar('Acc', acc, epoch)
                if acc > maxAcc:
                    maxAcc = acc
                    torch.save(model.state_dict(), '%s/%s_best_acc.pt' % (config['modelSaveDir'], config['model']))
                print(f'Epoch [{epoch + 1}/{config["epochs"]}], Loss: {loss.item():.4f}, Acc: {acc:.4f}')
            else:
                print(f'Epoch [{epoch + 1}/{config["epochs"]}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), '%s/%s_last.pt' % (config['modelSaveDir'], config['model']))

    if config['mode'] == 'test':

        model.eval()
        with torch.no_grad():
            tot_correct = 0
            tot_points = 0
            TN = 0
            TP = 0
            FN = 0
            FP = 0
            for batch_x, batch_y in dataLoader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x).transpose(2, 1)
                _, predicted = torch.max(output, 1)
                for i in range(predicted.shape[0]):
                    if predicted[i] == batch_y[i] and predicted[i] == 2:
                        TP += 1
                    elif predicted[i] == batch_y[i] and predicted[i] < 2:
                        TN += 1
                    elif predicted[i] != batch_y[i] and predicted[i] == 2:
                        FP += 1
                    else:
                        FN += 1
                correct = (predicted == batch_y).sum().item()
                tot_correct += correct
                tot_points += batch_y.numel()
            acc = tot_correct / tot_points
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2 * (precision * recall) / (precision + recall)
            print("Acc: %f" % acc)
            print("Precision: %f" % precision)
            print("Recall: %f" % recall)
            print("F1: %f" % F1)



