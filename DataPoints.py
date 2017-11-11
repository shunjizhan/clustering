import sys
import math


class DataPoints:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        self.isAssignedToCluster = False

    def __key(self):
        return (self.label, self.x, self.y)

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())

    @staticmethod
    def getMean(clusters, mean):
        for i in range(len(clusters)):
            cluster = clusters[i]
            size = len(cluster)
            sum_x, sum_y = 0.0, 0.0
            for p in cluster:
                sum_x += p.x
                sum_y += p.y
            mean[i][0] = sum_x / size
            mean[i][1] = sum_y / size

    @staticmethod
    def getStdDeviation(clusters, mean, stddev):
        for i in range(len(clusters)):
            cluster = clusters[i]
            m = mean[i]
            size = len(cluster)
            sum_x, sum_y = 0.0, 0.0
            for p in cluster:
                sum_x += (p.x - m[0]) ** 2
                sum_y += (p.y - m[1]) ** 2
            stddev[i][0] = math.sqrt(sum_x / size)
            stddev[i][1] = math.sqrt(sum_y / size)

    @staticmethod
    def getCovariance(clusters, mean, stddev, cov):
        for i in range(len(clusters)):
            cluster = clusters[i]
            m = mean[i]
            EX, EY = m[0], m[1]
            size = len(cluster)
            sum_xy = 0.0
            for p in cluster:
                sum_xy = (p.x - EX) * (p.y - EY)

            cov[i][0][0] = stddev[i][0] ** 2
            cov[i][1][1] = stddev[i][1] ** 2
            cov[i][0][1] = sum_xy / size
            cov[i][1][0] = sum_xy / size

    @staticmethod
    def getNMIMatrix(clusters, noOfLabels):
        nmiMatrix = [[0 for x in range(len(clusters) + 1)] for y in range(noOfLabels + 1)]
        clusterNo = 0
        for cluster in clusters:
            labelCounts = {}
            for point in cluster:
                if point.label not in labelCounts:
                    labelCounts[point.label] = 0
                labelCounts[point.label] += 1
            max = sys.maxint
            labelNo = 0
            labelTotal = 0
            labelCounts_sorted = sorted(labelCounts.iteritems(), key=lambda (k, v): (v, k), reverse=True)
            for label, val in labelCounts_sorted:
                nmiMatrix[label - 1][clusterNo] = labelCounts[label]
                labelTotal += labelCounts.get(label)
            nmiMatrix[noOfLabels][clusterNo] = labelTotal
            clusterNo += 1
            labelCounts.clear()

        # populate last col
        lastRowCol = 0
        for i in range(len(nmiMatrix) - 1):
            totalRow = 0
            for j in range(len(nmiMatrix[i]) - 1):
                totalRow += nmiMatrix[i][j]
            lastRowCol += totalRow
            nmiMatrix[i][len(clusters)] = totalRow
        nmiMatrix[noOfLabels][len(clusters)] = lastRowCol
        return nmiMatrix

    @staticmethod
    def calcNMI(nmiMatrix):
        # calculate I
        row = len(nmiMatrix)
        col = len(nmiMatrix[0])
        N = nmiMatrix[row - 1][col - 1]
        I = 0.0
        HOmega = 0.0
        HC = 0.0
        for i in range(row - 1):
            for j in range(col - 1):
                denominator = (float(nmiMatrix[i][col - 1]) * nmiMatrix[row - 1][j])
                if denominator == 0.0:
                    continue
                logPart = (float(N) * nmiMatrix[i][j]) / denominator
                if logPart == 0.0:
                    continue
                I += (nmiMatrix[i][j] / float(N)) * math.log(float(logPart))
                logPart1 = nmiMatrix[row - 1][j] / float(N)
                if logPart1 == 0.0:
                    continue
                HC += nmiMatrix[row - 1][j] / float(N) * math.log(float(logPart1))
            if float(N) == 0.0 or float(N) * math.log(nmiMatrix[i][col - 1] / float(N)) == 0.0:
                continue
            HOmega += nmiMatrix[i][col - 1] / float(N) * math.log(nmiMatrix[i][col - 1] / float(N))

        if math.sqrt(HC * HOmega) == 0.0:
            return 0.0
        return I / math.sqrt(HC * HOmega)

    @staticmethod
    def getNoOFLabels(dataSet):
        labels = set()
        for point in dataSet:
            labels.add(point.label)
        return len(labels)

    @staticmethod
    def writeToFile(noise, clusters, fileName):
        # write clusters to file for plotting
        f = open(fileName, 'w')
        for pt in noise:
            f.write(str(pt.x) + "," + str(pt.y) + ",0" + "\n")
        for w in range(len(clusters)):
            # print("Cluster " + str(w) + " size :" + str(len(clusters[w])))
            for point in clusters[w]:
                f.write(str(point.x) + "," + str(point.y) + "," + str((w + 1)) + "\n")
        f.close()
