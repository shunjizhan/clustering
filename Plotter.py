import random
import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def plot(clusters):
        for w in range(len(clusters)):
            print("Cluster " + str(w) + " size :" + str(len(clusters[w])))
            color = Plotter.randomColor()
            for p in clusters[w]:
                plt.plot(p.x, p.y, marker='o', c=color)
        plt.show()

    @staticmethod
    def randomColor():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return ('#%02X%02X%02X' % (r, g, b))
