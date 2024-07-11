import open3d as o3d
import numpy as np
from PointCloudClass import InitPointCloud
import argparse
import yaml
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="make dataset")
    parser.add_argument('--yaml-path', type=str, default='./config.yaml')
    args = parser.parse_args()

    with open(args.yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    totNum = config['endNum'] - config['startNum'] + 1

    for i in range(config['startNum'], config['endNum'] + 1):
        print("Now dealing with point cloud[%s/%s]" % (i - config['startNum'] + 1, totNum))
        inputPath = os.path.join(config['originDataDir'], config['fileName'] + str(i) + '.' + config['fileType'])
        outputDataPath = os.path.join(config['outputDataDir'], config['outputDataName'] + str(i) + ".txt")
        outputLablePath = os.path.join(config['outputLabelDir'], config['outputLabelName'] + str(i) + ".txt")

        pcd = o3d.io.read_point_cloud(inputPath, format=config['fileType'])
        data = np.asarray(pcd.points)
        data = data[:, :3]

        pointCloud = InitPointCloud(data, config['DOWNSAMPLE_SIZE'])
        pointCloud.RANSAC()
        if(config["singleHoleMode"] == 1):
            pointCloud.DBSCAN(config['DBSCAN_R'], config['DBSCAN_MIN_SAMPLE'], singleMode=True)
        else:
            pointCloud.DBSCAN(config['DBSCAN_R'], config['DBSCAN_MIN_SAMPLE'], singleMode=False)
        pointCloud.SaveData(outputDataPath, outputLablePath)
