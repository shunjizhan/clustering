from DataPoints import DataPoints
from Plotter import Plotter
import random
import sys
import math


def sqrt(n):
    return math.sqrt(n)


class Centroid:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not type(other) is type(self):
            return False
        if other is self:
            return True
        if other is None:
            return False
        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        return True

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def toString(self):
        return "Centroid [x=" + self.x + ", y=" + self.y + "]"

    def __str__(self):
        return self.toString()

    def __repr__(self):
        return self.toString()


class KMeans:
    def __init__(self):
        self.K = 0          # num of labels / clusters

    def main(self, args):
        seed = 71

        dataSet = self.readDataSet("dataset1.txt")
        self.K = DataPoints.getNoOFLabels(dataSet)
        random.Random(seed).shuffle(dataSet)
        self.kmeans(dataSet)

        print("")
        dataSet = self.readDataSet("dataset2.txt")
        self.K = DataPoints.getNoOFLabels(dataSet)
        random.Random(seed).shuffle(dataSet)
        self.kmeans(dataSet)

        print("")
        dataSet = self.readDataSet("dataset3.txt")
        self.K = DataPoints.getNoOFLabels(dataSet)
        random.Random(seed).shuffle(dataSet)
        self.kmeans(dataSet)

    def kmeans(self, dataSet):
        clusters = []
        k = 0
        while k < self.K:
            cluster = set()
            clusters.append(cluster)
            k += 1

        # Initially randomly assign points to clusters
        i = 0
        for point in dataSet:
            clusters[i % k].add(point)
            i += 1

        # calculate centroid for clusters
        centroids = []
        for j in range(self.K):
            centroids.append(self.getCentroid(clusters[j]))

        self.reassignClusters(dataSet, centroids, clusters)

        # continue till converge
        iteration = 0
        while True:
            iteration += 1

            # calculate centroid for clusters
            centroidsNew = []
            for j in range(self.K):
                centroidsNew.append(self.getCentroid(clusters[j]))

            # check onvergence
            isConverge = True
            for j in range(self.K):
                if centroidsNew[j] != centroids[j]:
                    isConverge = False
            if isConverge:
                break

            self.reassignClusters(dataSet, centroidsNew, clusters)
            for j in range(self.K):
                centroids[j] = centroidsNew[j]

        print("Iteration :" + str(iteration))
        # Calculate purity
        maxLabelCluster = []
        for j in range(self.K):
            maxLabelCluster.append(self.getMaxClusterLabel(clusters[j]))
        purity = 0.0
        for j in range(self.K):
            purity += maxLabelCluster[j]
        purity /= len(dataSet)
        print("Purity is :" + str(purity))

        noOfLabels = DataPoints.getNoOFLabels(dataSet)
        nmiMatrix = DataPoints.getNMIMatrix(clusters, noOfLabels)
        nmi = DataPoints.calcNMI(nmiMatrix)
        print("NMI :" + str(nmi))

        # plot the result
        Plotter.plot(clusters)

    def reassignClusters(self, dataSet, c, clusters):
        for j in range(self.K):
            clusters[j] = set()

        dist = [0.0 for x in range(self.K)]
        for point in dataSet:
            for i in range(self.K):
                dist[i] = self.getEuclideanDist(point.x, point.y, c[i].x, c[i].y)

            minIndex = self.getMinIndex(dist)
            clusters[minIndex].add(point)

    def getMinIndex(self, dist):
        min_ = sys.maxint
        minIndex = -1
        for i in range(len(dist)):
            if dist[i] < min_:
                min_ = dist[i]
                minIndex = i
        return minIndex

    def getEuclideanDist(self, x1, y1, x2, y2):
        dist = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
        return dist

    def getCentroid(self, cluster):
        x_sum, y_sum = 0.0, 0.0
        size = len(cluster)
        for p in cluster:
            x_sum += p.x
            y_sum += p.y

        return Centroid(x_sum / size, y_sum / size)

    @staticmethod
    def getMaxClusterLabel(cluster):
        labelCounts = {}
        for point in cluster:
            if point.label not in labelCounts:
                labelCounts[point.label] = 0
            labelCounts[point.label] += 1
        max = -sys.maxint - 1
        for label in labelCounts:
            if max < labelCounts[label]:
                max = labelCounts[label]
        return max

    @staticmethod
    def readDataSet(filePath):
        dataSet = []
        with open(filePath) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        for line in lines:
            points = line.split('\t')
            x = float(points[0])
            y = float(points[1])
            label = int(points[2])
            point = DataPoints(x, y, label)
            dataSet.append(point)
        return dataSet


if __name__ == "__main__":
    k = KMeans()
    k.main(None)
