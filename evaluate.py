from DataPoints import DataPoints
import sys
import random


class Evaluate:
    def __init__(self, predicts, reals):
        assert(len(predicts) == len(reals))
        self.predicts = predicts
        self.reals = reals
        self.K = 4                  # num of clusters
        self.N = len(predicts)      # num of points
        self.makeClusters()

    def makeClusters(self):
        clusters = [set() for i in range(self.K)]
        for i in range(self.N):
            true_label = reals[i] - 1
            p_label = predicts[i] - 1
            x, y = random.uniform(0.0, 5.0), random.uniform(0.0, 5.0)
            clusters[p_label].add(DataPoints(x, y, true_label))
        return clusters

    def purity(self):
        clusters = self.makeClusters()
        maxLabelCluster = []
        for j in range(self.K):
            maxLabelCluster.append(self.getMaxClusterLabel(clusters[j]))
        purity = 0.0
        for j in range(self.K):
            purity += maxLabelCluster[j]
        purity /= self.N
        return purity

    def getMaxClusterLabel(self, cluster):
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


if __name__ == '__main__':
    predicts = [3, 3, 1, 1, 1, 4, 3, 3, 4, 2, 4, 2, 1, 2, 3, 2, 1, 2, 4, 4]
    reals = [2, 2, 3, 3, 3, 4, 2, 2, 3, 1, 4, 1, 3, 1, 2, 1, 2, 1, 4, 4]
    e = Evaluate(predicts, reals)
    print e.purity()





