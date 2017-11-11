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
        self.clusters = clusters

    def purity(self):
        maxLabelCluster = []
        for j in range(self.K):
            maxLabelCluster.append(self.getMaxClusterLabel(self.clusters[j]))
        purity = 0.0
        for j in range(self.K):
            purity += maxLabelCluster[j]
        purity /= self.N
        print 'purity:', purity

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

    def P_R_F(self):
        TP, FN, FP, TN = 0.0, 0.0, 0.0, 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                same_cluster = (self.predicts[i] == self.predicts[j])
                same_class = (self.reals[i] == self.reals[j])

                if(same_class):
                    if(same_cluster):
                        TP += 1
                    else:
                        FN += 1
                else:           # different class
                    if(same_cluster):
                        FP += 1
                    else:
                        TN += 1

        assert(TP + FN + FP + TN == self.N * (self.N - 1) / 2)

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F = 2 * P * R / (P + R)

        print 'TP:', TP
        print 'TN:', TN
        print 'FP:', FP
        print 'FN:', FN
        print 'precision:', P
        print 'recall:', R
        print 'F-measure:', F

    def NMI(self):
        # nmiMatrix = DataPoints.getNMIMatrix(self.clusters, self.K)
        nmiMatrix = [
            [0, 1, 4, 0, 5],
            [5, 0, 0, 0, 5],
            [0, 5, 0, 0, 5],
            [0, 0, 1, 4, 5],
            [5, 6, 5, 4, 20],
        ]
        nmi = DataPoints.calcNMI(nmiMatrix)
        print 'NMI:', nmi


if __name__ == '__main__':
    reals = [3, 3, 1, 1, 1, 4, 3, 3, 4, 2, 4, 2, 1, 2, 3, 2, 1, 2, 4, 4]
    predicts = [2, 2, 3, 3, 3, 4, 2, 2, 3, 1, 4, 1, 3, 1, 2, 1, 2, 1, 4, 4]
    e = Evaluate(predicts, reals)
    e.purity()
    e.P_R_F()
    e.NMI()



