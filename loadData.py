import os
import open3d as o3d
import numpy as np
from tqdm import tqdm

# ModelNet40

def clean_data(data_path):
    with open(data_path, 'r') as file:
        line = file.readline()
        if len(line) < 4 :
            return

    with open(data_path, 'r') as file:
        lines = file.readlines()


    f = open(data_path, 'w')
    k = 0
    for line in lines:
        if k == 0:
            print(line[:3], file=f)
            print(line[3: -1], file=f)
        else :
            print(line[: -1], file=f)
        k += 1
    return

name_list = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
            'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
            'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
            'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa',
            'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
train_num_list = [626, 106, 515, 173, 572, 335, 64, 197, 889, 167, 79, 138, 200, 109, 200, 149,
                  171, 155, 145, 124, 149, 284, 465, 200, 88, 231, 240, 104, 115, 128, 680,
                  124, 90, 392, 163, 344, 267, 475, 87, 103]
test_num_test = [100, 50, 100, 20, 100, 100, 20, 100, 100, 20, 20, 20, 86, 20, 86, 20,
                  100, 100, 20, 20, 20, 100, 100, 86, 20, 100, 100, 20, 100, 20, 100,
                  20, 20, 100, 20, 100, 100, 100, 20, 20]
os.system('wget -P ./dataset http://modelnet.cs.princeton.edu/ModelNet40.zip')
os.system('unzip ./dataset/ModelNet40.zip -d ./dataset')
os.system('mv dataset/ModelNet40 dataset/ModelNet')
os.system('mkdir dataset/ModelNet40')
for i in tqdm(range(27, 40)):
    name = name_list[i]
    os.system('mkdir dataset/ModelNet40/' + name)
    os.system('mkdir dataset/ModelNet40/' + name + '/train')
    os.system('mkdir dataset/ModelNet40/' + name + '/test')
    for j in tqdm(range(train_num_list[i])):
        number_string = str(j + 1).zfill(4)
        path_obj = './dataset/ModelNet/' + name + '/train/' + name + '_' + number_string + '.off'
        clean_data(path_obj)
        mesh = o3d.io.read_triangle_mesh(path_obj)
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=1024)
        data = np.array(pcd.points)
        save_path = './dataset/ModelNet40/' + name + '/train/' + name + '_' + number_string + '.txt'
        np.savetxt(save_path, data)
    for j in tqdm(range(train_num_list[i], train_num_list[i] + test_num_test[i])):
        number_string = str(j + 1).zfill(4)
        path_obj = './dataset/ModelNet/' + name + '/test/' + name + '_' + number_string + '.off'
        clean_data(path_obj)
        mesh = o3d.io.read_triangle_mesh(path_obj)
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=1024)
        data = np.array(pcd.points)
        save_path = './dataset/ModelNet40/' + name + '/test/' + name + '_' + number_string + '.txt'
        np.savetxt(save_path, data)
os.system('rm -rf ./dataset/ModelNet')
os.system('rm ./dataset/ModelNet40.zip')
