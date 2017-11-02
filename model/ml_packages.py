# -*- coding = utf-8 =_=
__author__ = '15624959453@163.com'

import os
import sys
import re
# import graphviz
import numpy as np
from sklearn import datasets
from sklearn import svm, linear_model, tree, neighbors, neural_network
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

import matplotlib.pyplot as plt

# parameter for the option of machine learning model
class Parameter(object):
    cross_cv = 3 # K of k-fold method for cross validation

    SVM_gamma = 'auto' # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
    SVM_C = 0.0001 # Penalty parameter C of the error term.
    SVM_kernel = "linear" # Kernel type to be used in the algorithm, 'linear' and 'rbf' are valid

    LR_C = 1 # Penalty parameter C of the error term.
    LR_penalty = "l1" # Used to specify the norm used in the penalization
    LR_solver = "liblinear" # Algorithm to use in the optimization problem
    LR_tol = 0.001 # Tolerance for stopping criteria
    LR_Valid_ratio = 0.33 # Partion the dataset into train and validation, the ratio for validation.
    LR_coef_thresh = 0.2 # threshold used in relation analyse.

    KNN_k = 5 # K of k-means method

# Predict model base class:
# Define all the function and attributes which are essential for all method
# Parameters:
#     :dataset: patient gene mutation data
#     :labels: patient pathology type as labels
#     :titles: the mutation gene in order
#     :classifier: classifier instance of specific method 
# Methods:
#     set_params: reset the parameters of classifier
#     load_data: load data from hard disk into memory, get dataset and patient labels
#     train: train the model according to the given training data
#     predict: predict class labels for samples in X
#     evaluate: evaluate the accuracy on the dataset by cross validation method or set aside method
#     save_model: save the model attributes forever in hard disk
#     visualize: for some model like decision tree, the model structure can be reviewed in graph
class PredictModel(object):
    def __init__(self):
        # self.dataset = dataset
        self.dataset = []
        self.labels = []
        self.titles = []
        self.classifier = None

    def set_params(self, **kwargs):
        '''
        :param kwargs: the parameter of classifier model
        :return: None
        '''
        pass

    def load_data(self, fileName):
        '''
        :param fileName: the dataset file path
        :return: None
        '''
        matrix, labels, titles = [], [], []
        with open(fileName, "r") as f1:
            for lineno, line in enumerate(f1):
                data_array = line.strip("\n").split("\t")
                if lineno > 0:
                    if data_array[-1]:
                        labels.append(int(data_array[-1]))
                        matrix.append([float(x) for x in data_array[:-1]])
                else:
                    titles = data_array[:-1]
                #         matrix.append([int(x) for x in data_array[1:-3]])
                # else:
                #     titles = data_array[1:-3]
        self.dataset = np.array(matrix)
        self.titles = np.array(titles)
        self.labels = labels

    def train(self, train_set, train_labels):
        '''
        :param train_set: samples train data of sample features
        :param train_labels: sample train labels
        :return: None
        '''
        self.classifier.fit(train_set, train_labels)

    def predict(self, val_set):
        '''
        :param val_set: sample's feature
        :return: class labels predicted by samples in val_set
        '''
        return self.classifier.predict(val_set)

    def evaluate(self, dataset, labels):
        '''
        :param dataset: sample's feature dataset
        :param labels: class labels of real samples
        :return: mean accuracy of test data and labels
        '''
        pass

    def visualize(self):
        pass

    def save_model(self, fileName):
        pass


class LR(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = linear_model.LogisticRegression(
            C=Parameter.LR_C, penalty=Parameter.LR_penalty, tol=Parameter.LR_tol, solver=Parameter.LR_solver
        )

    # def load_data_temporary(self, fileName):
    #     matrix, labels, titles = [], [], []
    #     with open(fileName, "r") as f1:
    #         for lineno, line in enumerate(f1):
    #             data_array = line.strip("\n").split("\t")
    #             if lineno > 0:
    #                 if data_array[-1]:
    #                     labels.append(int(data_array[-1]))
    #                     matrix.append([float(x) for x in data_array[:-1]])
    #             else:
    #                 titles = data_array[:-1]
    #     self.dataset = np.array(matrix)
    #     self.titles = np.array([titles])
    #     self.labels = labels

    def load_data(self, fileName):
        matrix, labels, titles = [], [], []
        with open(fileName, "r") as f1:
            for lineno, line in enumerate(f1):
                data_array = line.strip("\n").split("\t")
                if lineno > 0:
                    if data_array[-1]:
                        labels.append(int(data_array[-1]))
                        matrix.append([int(x) for x in data_array[1:-3]])
                else:
                    titles = data_array[1:-3]
        self.dataset = np.array(matrix)
        self.titles = np.array([titles])
        self.labels = labels

    def train(self, train_set, train_labels):
        """
        Args:
        Returns:
        """
        self.classifier.fit(train_set, train_labels)

    def predict(self, val_set):
        predict_cls = self.classifier.predict(val_set)
        return predict_cls

    def evaluate_SetAside(self, dataset, labels):
        res = np.zeros((1, 10))
        for i in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=Parameter.LR_Valid_ratio)
            self.train(X_train, Y_train)
            # Y_predict = self.predict(X_test)
            acc = self.classifier.score(X_test, Y_test)
            res[0, i] = acc

            print("Loop {:d}\nAccuracy: {:4f}".format(i+1, acc))
        return res

    def evaluate_kfold(self, dataset, labels):
        scores = cross_val_score(self.classifier, dataset, labels, cv=Parameter.cross_cv)
        print("Accuracy of model LR: {:.4f}(+/- {:f})".format(scores.mean(), scores.std()*2))

    def tumor_analyse(self):
        try:
            clf_coef = self.classifier.coef_
        except:
            raise Exception("warning, classifier are supposed to fit some data.")
        valid_index = np.array(abs(clf_coef) > Parameter.LR_coef_thresh)
        print("Mutation express significantly:\n{}".format("\n".join(self.titles[valid_index])))


class SVM(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = svm.SVC(
            gamma = Parameter.SVM_gamma, C = Parameter.SVM_C, kernel=Parameter.SVM_kernel
        )

    def set_params(self, kernel_string, gamma, c):
        self.classifier.set_params(kernel=kernel_string, gamma=gamma, C=c)

    def load_data(self, fileName):
        # matrix, labels, titles = [], [], []
        # with open(fileName, "r") as f1:
        #     for lineno, line in enumerate(f1):
        #         data_array = line.strip("\n").split("\t")
        #         if lineno > 0:
        #             if data_array[-1]:
        #                 labels.append(int(data_array[-1]))
        #                 matrix.append([float(x) for x in data_array[:-1]])
        #         else:
        #             titles = data_array[:-1]
        # self.dataset = np.array(matrix)
        # self.titles = np.array([titles])
        # self.labels = labels

        matrix, labels, titles = [], [], []
        with open(fileName, "r") as f1:
            for lineno, line in enumerate(f1):
                data_array = line.strip("\n").split("\t")
                if lineno > 0:
                    if data_array[-1]:
                        labels.append(int(data_array[-1]))
                        matrix.append([int(x) for x in data_array[1:-3]])
                else:
                    titles = data_array[1:-3]
        self.dataset = np.array(matrix)
        self.titles = np.array([titles])
        self.labels = labels

    def train(self, train_set, labels):
        self.classifier.fit(train_set, labels)
        return self.classifier

    def predict(self, val_set):
        predicted = self.classifier.predict(val_set)
        # print("Classification report for classifier %s:\n%s\n"% (self.classifier, metrics.classification_report(expected, predicted)))
        # print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
        return predicted

    def evaluate(self, dataset, labels):
        res = np.zeros((1, 10))
        for i in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=Parameter.LR_Valid_ratio)
            self.train(X_train, Y_train)
            # Y_predict = self.predict(X_test)
            acc = self.classifier.score(X_test, Y_test)
            res[0, i] = acc

            print("---> Loop {:d}\nAccuracy: {:4f}".format(i+1, acc))
        print("Accuracy of model SVM: {:.4f}(+/- {:f})".format(res.mean(), res.std()))
        return res


class TreeClf(PredictModel):

    def __init__(self):
        PredictModel.__init__(self)
        # Scikit-learn uses an small optimised version of the CART algorithm
        self.classifier = tree.DecisionTreeClassifier()

    def evaluate(self, dataset, labels):
        scores = cross_val_score(self.classifier, dataset, labels, cv=Parameter.cross_cv)
        print(scores)
        print("Accuracy of model DecisionTree: {:.4f}(+/- {:f})".format(scores.mean(), scores.std() * 2))

    # seems that this function does not make sense.
    def save_model(self, fileName):
        dot_data = tree.export_graphviz(
            self.classifier, out_file=fileName, feature_names=self.titles, class_names=["0", "1"],
            filled=True, rounded=True, special_characters=True
        )
        # graph = graphviz.Source(dot_data)
        # graph.render("iris")
        return


class KNN(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = neighbors.KNeighborsClassifier(
            n_neighbors=Parameter.KNN_k, algorithm="auto"
        )

    def evaluate(self, dataset, labels):
        # scores = cross_val_score(self.classifier, dataset, labels, cv = Parameter.cross_cv)
        # print(scores)
        # print("Accuracy of model K_Nearst_Neighbor: {:.4f}(+/- {:f})".format(scores.mean(), scores.std() * 2))
        res = np.zeros((1, 10))
        for i in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=Parameter.LR_Valid_ratio)
            self.train(X_train, Y_train)
            # Y_predict = self.predict(X_test)
            acc = self.classifier.score(X_test, Y_test)
            res[0, i] = acc

            print("---> Loop {:d}\nAccuracy: {:4f}".format(i+1, acc))
        print("Accuracy of model K_NearstNeighbor: {:.4f}(+/- {:f})".format(res.mean(), res.std()))
        return res


class ShallowNetwork(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = neural_network.MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(20, 5), activation="relu"
        )

    def evaluate(self, dataset, labels):
        # scores = cross_val_score(self.classifier, dataset, labels, cv = Parameter.cross_cv)
        # print(scores)
        # print("Accuracy of model K_Nearst_Neighbor: {:.4f}(+/- {:f})".format(scores.mean(), scores.std() * 2))
        res = np.zeros((1, 10))
        for i in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=Parameter.LR_Valid_ratio)
            self.train(X_train, Y_train)
            Y_predict = self.predict(X_test)
            print(classification_report(Y_predict, Y_test))
            acc = self.classifier.score(X_test, Y_test)
            res[0, i] = acc

            print("---> Loop {:d}\nAccuracy: {:4f}".format(i+1, acc))
        print("Accuracy of model low layer nerual_network: {:.4f}(+/- {:f})".format(res.mean(), res.std()))
        return res


if __name__ == "__main__":
    # print(__doc__)
    # fileName = "D:\\公司项目_方文征\\胃癌检测项目\\Data\\突变鉴定\\mutation_for_1-52.table"
    fileName = "D:\\公司项目_方文征\\胃癌检测项目\\Code\\LogisticRegression-master\\data.txt" # Test data input

    ## 1. Logistic Regression Model
    # test = LR()
    # test.load_data(fileName)
    # print(test.dataset)
    # test.train(test.dataset, test.labels)
    # test.tumor_analyse()
    # test.evaluate_kfold(test.dataset, test.labels)
    # print("Total accuracy is: {:4f}".format(test.evaluate_SetAside(test.dataset, test.labels).mean()))

    ## 2. SVM Model
    # clf = SVM()
    # clf.load_data(fileName)
    # clf.evaluate(clf.dataset, clf.labels)

    ## 3. Decision Tree Model
    ## It seems like that tree model perform better than others
    # clf = TreeClf()
    # clf.load_data(fileName)
    # clf.evaluate(clf.dataset, clf.labels)
    # clf.train(clf.dataset, clf.labels)

    ## 4. KNN Model
    # clf = KNN()
    # clf.load_data(fileName)
    # clf.evaluate(clf.dataset, clf.labels)
    # # clf.train(clf.dataset, clf.labels)

    ## 5. Neural Network Model
    clf = ShallowNetwork()
    clf.load_data(fileName)
    clf.evaluate(clf.dataset, clf.labels)

