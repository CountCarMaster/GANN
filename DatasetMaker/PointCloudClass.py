import numpy as np

class PointCloud:
    def __init__(self, data:np.array):
        self.data = data
        self.pointNum = data.shape[0]
        self.channelNUm = data.shape[1]

    def DownSampling(self, sampleSize:int):


class InitPointCloud(PointCloud):
    def __init__(self, data:np.array):
        super().__init__(data)
        self.beClustered = False
        self.clusterResult = np.zeros(self.pointNum)


