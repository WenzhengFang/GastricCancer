# -*- coding = utf-8 =_=
__author__ = '15624959453@163.com'

import os
import sys
import re
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt


class PredictModel(object):
    def __init__(self):
        # self.dataset = dataset
        self.dataset = []

    def load_data(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def visualize(self):
        pass

class SVM(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)

    def load_data(self):
        iris = datasets.load_iris()
        digits = datasets.load_digits()
        self.dataset = list(zip(digits.images, digits.target))

    def visualize(self, num):
        for index, (image, label) in enumerate(self.dataset[:num]):
            plt.subplot(2, round(num/2.0), index+1)
            plt.axis("off")
            plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            plt.title("Training: %i" % label)
        plt.show()

    def train(self):
        pass

    def test(self):
        pass

if __name__ == "__main__":
    print(__doc__)
    t = SVM()
    t.load_data()
    t.visualize(11)


