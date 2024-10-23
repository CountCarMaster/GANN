import torch
import numpy as np
import json
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
import cv2
import h5py
from torch.utils.data import Dataset as Dataset2

class Dataset:
    def __init__(self, channel_num, batch_size):
        self.channel_num = channel_num
        self.batch_size = batch_size

class ModelNet40(Dataset):
    def __init__(self, batch_size, mode='all', channel_num=3):
        super(ModelNet40, self).__init__(channel_num, batch_size)
        self.categoriesNum = 40
        self.point_num = 1024
        self.mode = mode
        self.name_list = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
                     'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
                     'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
                     'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa',
                     'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.train_num_list = [626, 106, 515, 173, 572, 335, 64, 197, 889, 167, 79, 138, 200, 109, 200, 149,
                          171, 155, 145, 124, 149, 284, 465, 200, 88, 231, 240, 104, 115, 128, 680,
                          124, 90, 392, 163, 344, 267, 475, 87, 103]
        self.test_num_test = [100, 50, 100, 20, 100, 100, 20, 100, 100, 20, 20, 20, 86, 20, 86, 20,
                         100, 100, 20, 20, 20, 100, 100, 86, 20, 100, 100, 20, 100, 20, 100,
                         20, 20, 100, 20, 100, 100, 100, 20, 20]

    def Load(self):
        k_train = 0
        k_test = 0
        x_train = np.zeros([9843, self.point_num, self.channel_num])
        y_train = np.zeros(9843)
        x_test = np.zeros([2468, self.point_num, self.channel_num])
        y_test = np.zeros(2468)
        if self.mode == 'all':
            for i in range(40):
                name = self.name_list[i]
                for j in range(self.train_num_list[i]):
                    number_string = str(j + 1).zfill(4)
                    path = './dataset/ModelNet40/' + name + '/train/' + name + '_' + number_string + '.txt'
                    data_tmp = np.loadtxt(path)
                    x_train[k_train] = data_tmp
                    y_train[k_train] = i
                    k_train += 1
                for j in range(self.train_num_list[i], self.train_num_list[i] + self.test_num_test[i]):
                    number_string = str(j + 1).zfill(4)
                    path = './dataset/ModelNet40/' + name + '/test/' + name + '_' + number_string + '.txt'
                    data_tmp = np.loadtxt(path)
                    x_test[k_test] = data_tmp
                    y_test[k_test] = i
                    k_test += 1
            datasetTrain = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
            dataLoaderTrain = DataLoader(datasetTrain, batch_size=self.batch_size, shuffle=True, drop_last=True)
            datasetTest = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
            dataLoaderTest = DataLoader(datasetTest, batch_size=self.batch_size, shuffle=True, drop_last=True)
            return dataLoaderTrain, dataLoaderTest
        elif self.mode == 'train':
            for i in range(40):
                name = self.name_list[i]
                for j in range(self.train_num_list[i]):
                    number_string = str(j + 1).zfill(4)
                    path = './dataset/ModelNet40/' + name + '/train/' + name + '_' + number_string + '.txt'
                    data_tmp = np.loadtxt(path)
                    x_train[k_train] = data_tmp
                    y_train[k_train] = i
                    k_train += 1
            datasetTrain = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
            dataLoaderTrain = DataLoader(datasetTrain, batch_size=self.batch_size, shuffle=True, drop_last=True)
            return dataLoaderTrain
        elif self.mode == 'test':
            for i in range(40):
                name = self.name_list[i]
                for j in range(self.train_num_list[i], self.train_num_list[i] + self.test_num_test[i]):
                    number_string = str(j + 1).zfill(4)
                    path = './dataset/ModelNet40/' + name + '/test/' + name + '_' + number_string + '.txt'
                    data_tmp = np.loadtxt(path)
                    x_test[k_test] = data_tmp
                    y_test[k_test] = i
                    k_test += 1
            datasetTest = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
            dataLoaderTest = DataLoader(datasetTest, batch_size=self.batch_size, shuffle=True, drop_last=True)
            return dataLoaderTest

class ShapeNet(Dataset):
    def __init__(self, folderPath: str, batch_size, point_num=2048, mode='all', channel_num=3):
        super(ShapeNet, self).__init__(channel_num, batch_size)
        self.point_num = point_num
        self.mode = mode
        self.folderPath = folderPath

    def DownSampling(self, data: np.array, npoints):
        if data.shape[0] == npoints :
            return data
        elif data.shape[0] > npoints :
            idx = np.random.choice(data.shape[0], npoints, replace=False)
            data = data[idx]
            return data
        else :
            ans = data
            while ans.shape[0] + data.shape[0] < npoints:
                tmp = np.zeros([ans.shape[0] + data.shape[0], ans.shape[1]])
                tmp[:ans.shape[0], :] = ans
                tmp[ans.shape[0]:, :] = data
                ans = tmp
            idx = np.random.choice(data.shape[0], npoints - ans.shape[0], replace=False)
            tmp = np.zeros([npoints, data.shape[1]])
            tmp[:ans.shape[0], :] = ans
            tmp[ans.shape[0]:, :] = data[idx]
            return tmp

    def Load(self):
        trainPath = self.folderPath + '/shape_data/train_test_split/shuffled_train_file_list.json'
        testPath = self.folderPath + '/shape_data/train_test_split/shuffled_test_file_list.json'
        valPath = self.folderPath + '/shape_data/train_test_split/shuffled_val_file_list.json'
        with open(trainPath, 'r') as f:
            a = f.readline()
            a = a[2: -2]
            trainList = a.split('", "')
        with open(testPath, 'r') as f:
            a = f.readline()
            a = a[2: -2]
            testList = a.split('", "')
        with open(valPath, 'r') as f:
            a = f.readline()
            a = a[2: -2]
            valList = a.split('", "')
        x_train = np.zeros([len(trainList), self.point_num, self.channel_num])
        y_train = np.zeros([len(trainList), self.point_num])
        x_test = np.zeros([len(testList), self.point_num, self.channel_num])
        y_test = np.zeros([len(testList), self.point_num])
        x_val = np.zeros([len(valList), self.point_num, self.channel_num])
        y_val = np.zeros([len(valList), self.point_num])
        if self.mode == 'all' or self.mode == 'train':
            k = 0
            for name in trainList:
                xFileName = self.folderPath + '/' + name[: 20] + 'points/' + name[20:] + '.pts'
                yFileName = self.folderPath + '/' + name[: 20] + 'points_label/' + name[20:] + '.seg'
                x = np.loadtxt(xFileName)
                y = np.loadtxt(yFileName)
                tmp = np.zeros([x.shape[0], x.shape[1] + 1])
                tmp[:, :-1] = x
                tmp[:, -1] = y
                tmp = self.DownSampling(tmp, self.point_num)
                x_train[k, :, :] = tmp[:, :-1]
                y_train[k, :] = tmp[:, -1]
                k += 1
        if self.mode == 'all' or self.mode == 'test':
            k = 0
            for name in testList:
                xFileName = '../Dataset/' + name[: 20] + 'points/' + name[20:] + '.pts'
                yFileName = '../Dataset/' + name[: 20] + 'points_label/' + name[20:] + '.seg'
                x = np.loadtxt(xFileName)
                y = np.loadtxt(yFileName)
                tmp = np.zeros([x.shape[0], x.shape[1] + 1])
                tmp[:, :-1] = x
                tmp[:, -1] = y
                tmp = self.DownSampling(tmp, self.point_num)
                x_test[k, :, :] = tmp[:, :-1]
                y_test[k, :] = tmp[:, -1]
                k += 1
        if self.mode == 'all' or self.mode == 'val':
            k = 0
            for name in valList:
                xFileName = '../Dataset/' + name[: 20] + 'points/' + name[20:] + '.pts'
                yFileName = '../Dataset/' + name[: 20] + 'points_label/' + name[20:] + '.seg'
                x = np.loadtxt(xFileName)
                y = np.loadtxt(yFileName)
                tmp = np.zeros([x.shape[0], x.shape[1] + 1])
                tmp[:, :-1] = x
                tmp[:, -1] = y
                tmp = self.DownSampling(tmp, self.point_num)
                x_val[k, :, :] = tmp[:, :-1]
                y_val[k, :] = tmp[:, -1]
                k += 1
        if self.mode == 'all' :
            datasetTrain = TensorDataset(torch.tensor(x_train), torch.tensor(y_train - 1))
            dataLoaderTrain = DataLoader(datasetTrain, batch_size=self.batch_size, shuffle=True, drop_last=True)
            datasetTest = TensorDataset(torch.tensor(x_test), torch.tensor(y_test - 1))
            dataLoaderTest = DataLoader(datasetTest, batch_size=self.batch_size, shuffle=True, drop_last=True)
            datasetVal = TensorDataset(torch.tensor(x_val), torch.tensor(y_val - 1))
            dataLoaderVal = DataLoader(datasetVal, batch_size=self.batch_size, shuffle=True, drop_last=True)
            return dataLoaderTrain, dataLoaderTest, dataLoaderVal
        elif self.mode == 'train':
            datasetTrain = TensorDataset(torch.tensor(x_train), torch.tensor(y_train - 1))
            dataLoaderTrain = DataLoader(datasetTrain, batch_size=self.batch_size, shuffle=True, drop_last=True)
            return dataLoaderTrain
        elif self.mode == 'test':
            datasetTest = TensorDataset(torch.tensor(x_test), torch.tensor(y_test - 1))
            dataLoaderTest = DataLoader(datasetTest, batch_size=self.batch_size, shuffle=True, drop_last=True)
            return dataLoaderTest
        elif self.mode == 'val':
            datasetVal = TensorDataset(torch.tensor(x_val), torch.tensor(y_val - 1))
            dataLoaderVal = DataLoader(datasetVal, batch_size=self.batch_size, shuffle=True, drop_last=True)
            return dataLoaderVal

# path = '../Dataset/shape_data/train_test_split/shuffled_val_file_list.json'
# with open(path, 'r') as f:
#     a = f.readline()
#     a = a[2: -2]
#     lis = a.split('", "')
# minn = 10000
# maxx = -1
# for name in lis:
#     fileName = '../Dataset/' + name[: 20] + 'points_label/' + name[20:] + '.seg'
#     tmp = np.loadtxt(fileName)
#     if(np.min(tmp) < minn):
#         minn = min(tmp)
#     if(np.max(tmp) > maxx):
#         maxx = max(tmp)
# print(minn, maxx)

def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../Dataset', 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('indoor3d_sem_seg_hdf5_data', DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            sys.exit(0)
        else:
            zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('mv %s %s' % ('Stanford3dDataset_v1.2_Aligned_Version', DATA_DIR))
            os.system('rm %s' % (zippath))

def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../Dataset', 'data')
    if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
        os.system('python ./Dataset/prepare_data/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')):
        os.system('python ./Dataset/prepare_data/gen_indoor3d_h5.py')

def load_color_semseg():
    colors = []
    labels = []
    f = open("./Dataset/prepare_data/meta/semseg_colors.txt")
    for line in json.load(f):
        colors.append(line['color'])
        labels.append(line['label'])
    semseg_colors = np.array(colors)
    semseg_colors = semseg_colors[:, [2, 1, 0]]
    partseg_labels = np.array(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 1500
    img = np.zeros((500, img_size, 3), dtype="uint8")
    cv2.rectangle(img, (0, 0), (img_size, 750), [255, 255, 255], thickness=-1)
    color_size = 64
    color_index = 0
    label_index = 0
    row_index = 16
    for _ in range(0, img_size):
        column_index = 32
        for _ in range(0, img_size):
            color = semseg_colors[color_index]
            label = partseg_labels[label_index]
            length = len(str(label))
            cv2.rectangle(img, (column_index, row_index), (column_index + color_size, row_index + color_size),
                          color=(int(color[0]), int(color[1]), int(color[2])), thickness=-1)
            img = cv2.putText(img, label, (column_index + int(color_size * 1.15), row_index + int(color_size / 2)),
                              font,
                              0.7, (0, 0, 0), 2)
            column_index = column_index + 200
            color_index = color_index + 1
            label_index = label_index + 1
            if color_index >= 13:
                cv2.imwrite("prepare_data/meta/semseg_colors.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                return np.array(colors)
            elif (column_index >= 1280):
                break
        row_index = row_index + int(color_size * 1.3)
        if (row_index >= img_size):
            break

def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../Dataset', 'data')
    download_S3DIS()
    prepare_test_data_semseg()
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg

class S3DIS(Dataset2):
    def __init__(self, num_points=2048, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition
        self.semseg_colors = load_color_semseg()

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]

class S3DISFinal(Dataset):
    def __init__(self, batch_size, point_num=2048, mode='all', channel_num=9):
        super(S3DISFinal, self).__init__(channel_num, batch_size)
        self.point_num = point_num
        self.mode = mode
    def Load(self):
        if self.mode == 'all':
            train_loader = DataLoader(S3DIS(partition='train', num_points=self.point_num), num_workers=8, batch_size=self.batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(S3DIS(partition='test', num_points=self.point_num), num_workers=8, batch_size=self.batch_size, shuffle=True, drop_last=True)
            return train_loader, test_loader

        elif self.mode == 'train':
            train_loader = DataLoader(S3DIS(partition='train', num_points=self.point_num), num_workers=8,
                                      batch_size=self.batch_size, shuffle=True, drop_last=True)
            return train_loader

        elif self.mode == 'test':
            test_loader = DataLoader(S3DIS(partition='test', num_points=self.point_num), num_workers=8,
                                     batch_size=self.batch_size, shuffle=True, drop_last=True)
            return test_loader

# a = S3DISFinal(batch_size=4, point_num=2048, mode='train')
# b = a.Load()
# for x, y in b:
#     print(x.shape, y.shape)
#     print(x)
#     print(y)