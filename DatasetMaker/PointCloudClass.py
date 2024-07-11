import numpy as np
from ErrorClass import ProcessError, KeyError
from queue import Queue
from scipy.spatial import cKDTree
import open3d as o3d

class PointCloud:
    def __init__(self, data:np.array):
        self.data = data
        self.pointNum = data.shape[0]
        self.channelNum = data.shape[1]
        self.index = np.zeros(self.pointNum, dtype=np.int8)
        for i in range(self.pointNum):
            self.index[i] = i

    def DownSampling(self, sampleSize:int):
        """
        downsample the data from pointNum to sampleSize by FPS, change the data.
        :param sampleSize: pointNum of the downsampled data.
        :return: None.
        """
        centroids = np.zeros((sampleSize,))
        distance = np.ones((self.pointNum,)) * 1e10
        farthest = np.random.randint(0, self.pointNum)
        for i in range(sampleSize):
            centroids[i] = farthest
            centroid = self.data[farthest, :]
            dist = np.sum((self.data - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        self.data = self.data[centroids.astype(np.int32)]
        self.pointNum = sampleSize
        self.index = np.zeros(self.pointNum, dtype=np.int8)
        for i in range(self.pointNum):
            self.index[i] = i
        return

    def Show(self):
        """
        show the point cloud data by Open3D.
        :return: None.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.data)
        o3d.visualization.draw_geometries([pcd])
        return


class InitPointCloud(PointCloud):
    def __init__(self, data:np.array, downSampleSize=None):
        super().__init__(data)
        self.beClustered = False
        self.beModeled = False
        if(downSampleSize != None):
            self.DownSampling(downSampleSize)
        self.clusterResult = np.zeros(self.pointNum, dtype=np.int8)

    def Dist(self, A, B, C, D, point:np.array):
        """
        calculate the distance between the point and the surface.
        :param A, B, C, D: parameters of the surface.
        :param point: np.array([x, y, z]).
        :return: distance.
        """
        return abs(A * point[0] + B * point[1] + C * point[2] + D) / np.sqrt(A * A + B * B + C * C)

    def RANSAC(self, e=0.7, s=3, p=0.99, tau=0.01):
        """
        get the ground point of the point cloud using RANSAC.
        ground point in self.clusterResult will be 1.
        :param e: the probability out of the surface.
        :param s: the number of point to decide the surface.
        :param p: correct probability.
        :param tau: the shortest distance threshold.
        :return: normal vector of the surface.
        """
        iterateNum = int(np.log(1 - p) / np.log(1 - np.power((1 - e), s)))
        bestA, bestB, bestC, bestD = 0, 0, 0, 0
        bestInliner = -1
        bestIdx = None
        for i in range(iterateNum):
            randomIndex = np.random.choice(self.pointNum, s, replace=False)
            point_0 = self.data[randomIndex[0]]
            point_1 = self.data[randomIndex[1]]
            point_2 = self.data[randomIndex[2]]
            vector_0 = point_1 - point_0
            vector_1 = point_2 - point_1
            normal = np.cross(vector_0, vector_1)
            A, B, C = normal[0], normal[1], normal[2]
            D = -np.dot(normal, point_0)

            inliner = 0
            idx = np.zeros(self.pointNum)
            for j in range(self.pointNum):
                if self.Dist(A, B, C, D, self.data[j]) < tau:
                    inliner += 1
                    idx[j] = 1
            if inliner > bestInliner:
                bestIdx = idx
                bestInliner = inliner
                bestA, bestB, bestC, bestD = A, B, C, D
        for i in range(self.pointNum):
            self.clusterResult[i] = bestIdx[i]
        bestNormal = np.array([bestA, bestB, bestC])
        self.beModeled = True

        return bestNormal

    def DBSCAN(self, r, min_sample, singleMode=False):
        """
        clustering the data by DBSCAN algorithm. the clustering result will be saved as self.clusterResult
        clusterResult == 0 -> noise
        clusterResult == 1 -> ground
        clusterResult == 2 -> hole
        :param r: the radius for every single iterate.
        :param min_sample: the minimum number of samples.
        :return: None.
        """
        if(self.beModeled == False):
            raise ProcessError('You have to model the point cloud first. '
                               'Try InitPointCloud.RANSAC() then use this function')
        tmpData = self.data[self.clusterResult == 0]
        tmpIndex = self.index[self.clusterResult == 0]
        vis = np.zeros([tmpData.shape[0]])
        clusterNum = 0
        dataQueue = Queue()
        kdtree = cKDTree(tmpData)
        clusterAns = np.zeros([tmpData.shape[0]])

        # fit
        while(1):
            if np.sum(vis) == tmpData.shape[0]:
                break
            clusterNum += 1
            unvisitedData = np.where(vis == 0)[0]
            targetPoint = np.random.choice(unvisitedData, 1, replace=False)[0]
            dataQueue.put(targetPoint)
            vis[targetPoint] = 1
            while(dataQueue.empty() == False):
                nowPoint = dataQueue.get()
                neighbors = kdtree.query_ball_point(tmpData[nowPoint], r=r)
                if(len(neighbors) < min_sample):
                    clusterAns[nowPoint] = 0
                    vis[nowPoint] = 1
                    clusterNum -= 1
                else:
                    clusterAns[nowPoint] = clusterNum
                    for i in neighbors:
                        if(vis[i] == 1):
                            continue
                        dataQueue.put(i)
                        vis[i] = 1

        for i in range(clusterNum):
            cluster = PointCloud(tmpData[clusterAns == i + 1])
            cluster.index = tmpIndex[clusterAns == i + 1]
            print("Now showing the [%s/%s] cluster, is that hole?(Y/n)" % (i + 1, clusterNum))
            cluster.Show()
            ans = input('Y/n: ')
            if(ans == 'N' or ans == "n"):
                continue
            elif(ans == 'Y' or ans == "y"):
                for indx in cluster.index:
                    self.clusterResult[indx] = 2
                if(singleMode == True):
                    break
            else:
                raise KeyError("Please enter Y or n")

    def SaveData(self, pathX, pathY):
        """
        save the result, include the downsampled data as x, and clusterResult as y.
        :param pathX: save path of data.
        :param pathY: save path of label.
        :return: None.
        """
        np.savetxt(pathX, self.data)
        np.savetxt(pathY, self.clusterResult, fmt='%d')
        return