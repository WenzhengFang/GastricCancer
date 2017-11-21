# -*- coding = utf-8 =_=
__author__ = '15624959453@163.com'

import os
import sys
import re
#from Tools.IO import FileIO, DirIO
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
sys.path.append("../data_process/")

import numpy as np
#from xgboost.sklearn import XGBClassifier
from sklearn import svm, linear_model, tree, neighbors, neural_network, ensemble, naive_bayes
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from Dimension_reduction import Feature_selection
import matplotlib.pyplot as plt
import pydotplus

class Parameter(object):
    infoGain_thresh = 0.075 #mutual information
    cross_cv = 3  # k-fold
    top_gene_counts = 15 #how many first mutual 

    SVM_gamma = 0.001  #gamma for SVM
    SVM_C = 0.076  
    SVM_kernel = "linear"
    SVM_coef = 0.54
    SVM_degree = 3

    RF_n = 50
    RF_minSplit = 2
    RF_randState = 0

    GBDT_n = 100
    GBDT_learnRate = 0.1

    LR_C = 0.02
    LR_penalty = "l2"
    LR_solver = "liblinear" # Algorithm to use in the optimization problem ["liblinear",""]
    LR_tol = 0.001 # Tolerance for stopping criteria
    LR_Valid_ratio = 0.3 # Partion the dataset into train and validation, the ratio for validation.
    LR_coef_thresh = 0.2 # threshold used in relation analyse.

    KNN_k = 5

    Bayes_alpha = 0.1
    Bayes_probThresh = 0.003

class PredictModel(object):
    def __init__(self):
        # Feature_selection.__init__(self)
        # self.dataset = dataset
        self.dataset = []
        self.labels = []
        self.titles = []
        self.classifier = None

    def feature_reduction(self, dataset, titles, labels, thresh, feat_impace_file=None):
        """
            Select feature by information gain method.
        """
        feature_oper = Feature_selection()
        if not feat_impace_file:
            new_dataset, new_titles = feature_oper.feature_select(dataset, titles, labels, thresh)
        else:
            new_dataset, new_titles = feature_oper.sec_feature_select(dataset, titles, feat_impace_file, Parameter.top_gene_counts)
        return new_dataset, new_titles

    def set_params(self, **kwargs):
        pass

    def load_data(self, fileName):
        matrix, labels, titles = [], [], []
        with open(fileName, "r") as f1:
            for lineno, line in enumerate(f1):
                data_array = line.strip("\n").split("\t")
                if lineno > 0:
                    if data_array[-1]:
                        labels.append(int(data_array[-1]))
                #         matrix.append([float(x) for x in data_array[:-1]])
                # else:
                #     titles = data_array[:-1]
                        matrix.append([int(x) for x in data_array[1:-3]])
                else:
                    titles = data_array[1:-3]
        self.dataset = np.array(matrix)
        self.titles = np.array(titles)
        self.labels = labels

    def param_compare(self, tuned_parameters, dataset, labels, scores_method):
        X_train, X_test, y_train, y_test = train_test_split(
            dataset, labels, test_size=Parameter.LR_Valid_ratio, random_state=0
        )
        for score in scores_method:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(
                estimator=self.classifier, param_grid=tuned_parameters, cv=Parameter.cross_cv, scoring=score
            )
            clf.fit(X_train, y_train)
            print("Grid scores on development set:")
            print()
            #for params, mean_score, scores in clf.grid_scores_:
            #    print("%0.3f (+/-%0.03f) for %r"
            #          % (mean_score, scores.std() * 2, params))
            print()

            print("Best parameters and scores set found on development set by {}:".format(score))
            print()
            print(clf.best_params_)
            print(clf.best_score_)
            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

    def train(self, train_set, train_labels):
        self.classifier.fit(train_set, train_labels)

    def predict(self, val_set):
        return self.classifier.predict(val_set)

    def evaluate(self, dataset, labels):
        pass

    def visualize(self, **kwargs):
        pass

    def feature_select(self, **kwargs):
        pass

    def save_model(self, fileName):
        pass


class LR(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = linear_model.LogisticRegression(
            C=Parameter.LR_C, penalty=Parameter.LR_penalty, tol=Parameter.LR_tol, solver=Parameter.LR_solver
        )

    def set_params(self, C_up, penalty_up):
        self.classifier.set_params(C=C_up, penalty=penalty_up)

    def train(self, train_set, train_labels):
        """
        Args:
        Returns:
        """
        self.classifier.fit(train_set, train_labels)

    def predict(self, val_set):
        predict_cls = self.classifier.predict(val_set)
        return predict_cls

    def evaluate(self, dataset, labels):
        #scores = cross_val_score(self.classifier, dataset, labels, cv=Parameter.cross_cv)
        #print(scores)
        #print("\tAccuracy of model LR: {:.4f}(+/- {:f})".format(scores.mean(), scores.std() * 2))
        #return scores
        res = np.zeros((1, 10))
        for i in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=Parameter.LR_Valid_ratio)
            self.train(X_train, Y_train)
            # Y_predict = self.predict(X_test)
            acc = self.classifier.score(X_test, Y_test)
            res[0, i] = acc
            print("Loop {:d}\nAccuracy: {:4f}".format(i+1, acc))
        print("Accuracy of model LR: {:.4f}(+/- {:f})".format(res.mean(), res.std()*2))
        return res

    def tumor_analyse(self):
        try:
            clf_coef = self.classifier.coef_
        except:
            raise Exception("warning, classifier are supposed to fit some data.")
        valid_index = np.array(abs(clf_coef) > Parameter.LR_coef_thresh)
        print("Mutation express significantly:\n{}".format("\n".join(self.titles[valid_index])))

    def visualize(self):
        y, x = [[], []], []
        external = 1000.0
        plt.figure()
        self.dataset, self.titles = self.feature_reduction(self.dataset, self.titles, self.labels, Parameter.infoGain_thresh)
        for no, p in enumerate(["l1", "l2"]):
            Parameter.LR_penalty = p
            for i in range(1, 1000):
                Parameter.LR_C = i / external
                if no == 0:
                    x.append(Parameter.LR_C)
                self.set_params(C_up=Parameter.LR_C, penalty_up=Parameter.LR_penalty)
                print("-> Parameter status:\tC({0:f})\tpenalty({1:s})".format(Parameter.LR_C, Parameter.LR_penalty))
                score = self.evaluate(self.dataset, self.labels).mean()
                y[no].append(score)
        plt.plot(x, y[0], label = "curve for l1 penalty")
        plt.plot(x, y[1], label = "curve for l2 penalty")
        plt.title("plot for the relationship of C and prediction accuracy")
        plt.legend(loc = "upper right")
        plt.xlabel("C/{:f}".format(1/external))
        plt.ylabel("Accuracy/1")
        plt.grid(x)
        plt.show()

    def feature_select(self):
        coef_array = [(cf, i) for i, cf in enumerate(self.classifier.coef_[0])]
        coef_array.sort(key=lambda x: abs(x[0]), reverse=True)
        return ["LR"] + ["{:s}({:.6f})".format(self.titles[ind[1]], ind[0]) for ind in coef_array]


class SVM(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = svm.SVC(
            gamma = Parameter.SVM_gamma, C = Parameter.SVM_C, kernel=Parameter.SVM_kernel, coef0=Parameter.SVM_coef, degree=Parameter.SVM_degree
        )

    def set_params(self, kernel_string, gamma, c):
        self.classifier.set_params(kernel=kernel_string, gamma=gamma, C=c)

    def evaluate(self, dataset, labels):
        #scores = cross_val_score(self.classifier, dataset, labels, cv=Parameter.cross_cv)
        #print("Accuracy of model SVM: {:.4f}(+/- {:f})".format(scores.mean(), scores.std() * 2))
        #return scores
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

    def visualize(self):
        y, x = [], []
        external = 1.0
        plt.figure()
        self.dataset, self.titles = self.feature_reduction(self.dataset, self.titles, self.labels, Parameter.infoGain_thresh)
        for i in range(1, 1001):
            Parameter.SVM_C = i / external
            x.append(Parameter.SVM_C)
            self.set_params(kernel_string=Parameter.SVM_kernel, gamma=Parameter.SVM_gamma, c=Parameter.SVM_C)
            print("-> Parameter status:\tC({0:f})".format(Parameter.SVM_C))
            score = self.evaluate(self.dataset, self.labels).mean()
            y.append(score)
        plt.plot(x, y)
        plt.title("plot for the relationship of C and prediction accuracy in linear kernel")
        plt.legend(loc = "upper right")
        plt.xlabel("C/{:f}".format(1/external))
        plt.ylabel("Accuracy/1")
        plt.grid(x)
        plt.show()

    def feature_select(self):
        coef_array = [(cf, i) for i, cf in enumerate(self.classifier.coef_[0])]
        coef_array.sort(key=lambda x: abs(x[0]), reverse=True)
        return ["SVM"] + ["{:s}({:.6f})".format(self.titles[ind[1]], ind[0]) for ind in coef_array]


class TreeClf(PredictModel):

    def __init__(self):
        PredictModel.__init__(self)
        # Scikit-learn uses an small optimised version of the CART algorithm
        self.classifier = tree.DecisionTreeClassifier(max_leaf_nodes=28, min_impurity_split=0.074)

    def evaluate(self, dataset, labels):
        # scores = cross_val_score(self.classifier, dataset, labels, cv=Parameter.cross_cv)
        # print(scores)
        # print("Accuracy of model DecisionTree: {:.4f}(+/- {:f})".format(scores.mean(), scores.std() * 2))
        # return scores

        res = np.zeros((2, 10))
        for i in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=Parameter.LR_Valid_ratio)
            self.train(X_train, Y_train)
            # Y_predict = self.predict(X_test)
            val_acc = self.classifier.score(X_test, Y_test)
            train_acc = self.classifier.score(X_train, Y_train)
            res[:, i] = [val_acc, train_acc]

            print("---> Loop {:d}\n\tVal_Accuracy: {:.4f}\tTrain_Accuracy: {:.4f}".format(i+1, val_acc, train_acc))
        print("Val_Accuracy and Train_Accuracy of model Decision Tree:\n\t{:.4f}(+/- {:f})\t{:.4f}(+/- {:f})".format(
            res[0, :].mean(), res[0, :].std(), res[1, :].mean(), res[1, :].std())
        )
        return res

    # seems that this function does not make sense.
    def visualize(self, fileName):
        dot_data = tree.export_graphviz(self.classifier, out_file=None, feature_names=clf.titles, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(fileName)
        return


class Bayes(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = naive_bayes.BernoulliNB(
            alpha=Parameter.Bayes_alpha, class_prior = None
        )

    def evaluate(self, dataset, labels):
        scores = cross_val_score(self.classifier, dataset, labels, cv=Parameter.cross_cv)
        print(scores)
        print("Accuracy of model Bayes: {:.4f}(+/- {:f})".format(scores.mean(), scores.std() * 2))

    def feature_select(self):
        log_prob_pos = self.classifier.feature_log_prob_
        real_prob_pos = np.exp(log_prob_pos)
        label_prob_pos = sum(self.labels) / len(self.labels)
        label_prob_neg = 1 - label_prob_pos

        real_ratio, buffer = [], []
        for i in range(real_prob_pos.shape[1]):
            max_prob, min_prob = sorted([real_prob_pos[0][i], real_prob_pos[1][i]], reverse=True)
            if max_prob > 0.8:
                max_prob, min_prob = 1 - min_prob, 1 - max_prob
            if min_prob > Parameter.Bayes_probThresh:
                real_ratio.append((max_prob / min_prob, i))
            else:
                buffer.append((max_prob, i))
        real_ratio.sort(key=lambda x: x[0], reverse=True)
        buffer.sort(key=lambda x: x[0], reverse=True)
        content = ["BNBayes"] + ["{}({:.6f})".format(self.titles[x[1]], x[0]) for x in real_ratio + buffer]
        return content


class Xgboost(PredictModel):
    '''
        waiting for that the package installed successfully.
    '''
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = XGBClassifier(
            learning_rate = 0.1, n_estimators=140, max_depth=5, min_child_weight=1,
            gamma=0, subsample=0.8, colsample_bytree=0.8, objective="binary:logistic"
        )

    def evaluate(self, dataset, labels):
        scores = cross_val_score(self.classifier, dataset, labels, cv=Parameter.cross_cv)
        print(scores)
        print("Accuracy of model Xgboost: {:.4f}(+/- {:f})".format(scores.mean(), scores.std() * 2))


class RandomForest(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = ensemble.RandomForestClassifier(
            n_estimators=Parameter.RF_n, max_depth=None, min_samples_split=Parameter.RF_minSplit,
            random_state=Parameter.RF_randState, max_features="auto"
        )

    def evaluate(self, dataset, labels):
        res = np.zeros((1, 10))
        for i in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=Parameter.LR_Valid_ratio)
            self.train(X_train, Y_train)
            # Y_predict = self.predict(X_test)
            acc = self.classifier.score(X_test, Y_test)
            res[0, i] = acc

            print("---> Loop {:d}\nAccuracy: {:4f}".format(i+1, acc))
        print("Accuracy of model Random Forest: {:.4f}(+/- {:f})".format(res.mean(), res.std()))
        return res


class GBDT(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = ensemble.GradientBoostingClassifier(
            n_estimators=Parameter.GBDT_n, learning_rate=Parameter.GBDT_learnRate, max_depth=1, random_state=0
        )
        # self.classifier.fit([[1, 2]], [1])

    def evaluate(self, dataset, labels):
        scores = cross_val_score(self.classifier, dataset, labels, cv=Parameter.cross_cv)
        print(scores)
        print("Accuracy of model DecisionTree: {:.4f}(+/- {:f})".format(scores.mean(), scores.std() * 2))
        # res = np.zeros((1, 10))
        # for i in range(10):
        #     X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=Parameter.LR_Valid_ratio)
        #     self.train(X_train, Y_train)
        #     # Y_predict = self.predict(X_test)
        #     acc = self.classifier.score(X_test, Y_test)
        #     res[0, i] = acc
        #
        #     print("---> Loop {:d}\nAccuracy: {:4f}".format(i+1, acc))
        # print("Accuracy of model GBDT: {:.4f}(+/- {:f})".format(res.mean(), res.std()))
        # return res


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

            # print("---> Loop {:d}\nAccuracy: {:4f}".format(i+1, acc))
        print("Accuracy of model K_NearstNeighbor: {:.4f}(+/- {:f})".format(res.mean(), res.std()))
        return res


class ShallowNetwork(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = neural_network.MLPClassifier(
            solver="lbfgs", alpha=1e-4, hidden_layer_sizes=(20, 5), activation="relu"
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


def feature_impact_estimate(mut_analyse_file, mut_impact_file, threshold, top_counts, model_array):
    ins_array = [eval("{0}()".format(model_str)) for model_str in model_array]
    for instance in ins_array:
        clf = instance
        clf.load_data(mut_analyse_file)

__end__ = "yes"


if __name__ == "__main__":
    # print(__doc__)
    fileName = "/Users/classxiaoli/Desktop/wenzheng/mutation_for_1-78.table"
    threshold = 0.075
    top_counts = 35
    matrix = []
    mutation_sort_file = "/Users/classxiaoli/Desktop/wenzheng/mutation_importance_by_ml_top{:d}.txt".format(top_counts)
    # fileName = "D:\\公司项目_方文征\\胃癌检测项目\\Code\\LogisticRegression-master\\data.txt" # Test data input

    # 1. Logistic Regression Model
    test = LR()
    test.load_data(fileName)
    tuned_parameters = {
        # "penalty":["l2", "l1"],
        "penalty":["l2"],
        "C": [i/1000.0 for i in range(1, 3001)] + [i for i in range(4, 1001)],
        # "C": [0.129]
    }
    scores = ["accuracy"]
    test.dataset, test.titles = test.feature_reduction(test.dataset, test.titles, test.labels, threshold, mutation_sort_file)
    #test.dataset, test.titles = test.feature_reduction(test.dataset, test.titles, test.labels, threshold)
    #test.param_compare(tuned_parameters, test.dataset, test.labels, scores)
    test.train(test.dataset, test.labels)
    test.evaluate(test.dataset, test.labels)
    #matrix.append(test.feature_select())

    ## 2. SVM Model
    #clf = SVM()
    #clf.load_data(fileName)
    #tuned_parameters = {
    #    # "gamma":[i/1000.0 for i in range(1, 1001)],
    #    "gamma":[i/100 for i in range(1, 101)],
    #    # "C": [i/1000.0 for i in range(1, 1001)] + [i for i in range(2, 5001)],
    #    "C": [i/10.0 for i in range(1, 101)],
    #    # "kernel": ["rbf", "linear", "poly", "sigmoid"],
    #    "kernel": ["sigmoid"],
    #    "degree": [3],
    #    "coef0": [i/10 for i in range(1, 101)]
    #}
    #scores_method = ["accuracy"]
    #clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, threshold)
    ##clf.param_compare(tuned_parameters, clf.dataset, clf.labels, scores_method)
    ## print("With coefficient of SVM: {}; ".format(Parameter.SVM_C))
    #clf.evaluate(clf.dataset, clf.labels)
    #clf.train(clf.dataset, clf.labels)
    #matrix.append(clf.feature_select())

    ## 3. Decision Tree Model
    # tuned_parameters = {
    #     # "max_depth":[i for i in range(35, 36)],
    #     "max_leaf_nodes": [i for i in range(20, 100)],
    #     "min_impurity_split": [i/1000 for i in range(1, 101)]
    # }
    # scores_method = ["accuracy"]
    # clf = TreeClf()
    # clf.load_data(fileName)
    # clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, threshold)
    # # clf.param_compare(tuned_parameters, clf.dataset, clf.labels, scores_method)
    # clf.evaluate(clf.dataset, clf.labels)
    # # clf.train(clf.dataset, clf.labels)
    # # clf.visualize("D:\\公司项目_方文征\\胃癌检测项目\\Data\\tree.pdf")

    ## 4. KNN Model
    # tuned_parameters = {
    #     "n_neighbors":[i for i in range(1, 21)]
    # }
    # scores_method = ["accuracy"]
    # clf = KNN()
    # clf.load_data(fileName)
    # clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, threshold)
    # clf.param_compare(tuned_parameters, clf.dataset, clf.labels, scores_method)
    # clf.evaluate(clf.dataset, clf.labels)
    # clf.train(clf.dataset, clf.labels)

    ## 5. Neural Network Model
    #tuned_parameters = {
    #    "alpha": [i/1000.0 for i in range(1000)],
    #    "hidden_layer_sizes": [(10, 5, 2)]
    #}
    #scores_method = ["accuracy"]
    #clf = ShallowNetwork()
    #clf.load_data(fileName)
    #clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, threshold)
    #clf.param_compare(tuned_parameters, clf.dataset, clf.labels, scores_method)
    # clf.evaluate(clf.dataset, clf.labels)

    ## 6. Random Forest Model
    # tuned_parameters = {
    #     "max_features":[i/100 for i in range(1, 11)],
    #     "n_estimators": [i for i in range(1, 50)]
    # }
    # scores_method = ["accuracy"]
    # clf = RandomForest()
    # clf.load_data(fileName)
    # clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, threshold)
    # clf.param_compare(tuned_parameters, clf.dataset, clf.labels, scores_method)
    # clf.evaluate(clf.dataset, clf.labels)

    ## 7. Gradient Boosting Desicision Tree
    # tuned_parameters = {
    #     "loss": ["deviance", "exponential"],
    #     "learning_rate": [i/10.0 for i in range(1, 2)],
    #     "n_estimators": [i for i in range(20, 200)],
    #     "max_features": [i / 20 for i in range(1, 21)]
    # }
    # scores_method = ["accuracy"]
    # clf = GBDT()
    # clf.load_data(fileName)
    # clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, threshold)
    # clf.param_compare(tuned_parameters, clf.dataset, clf.labels, scores_method)

    ## 8. Naive Bayes
    # tuned_parameters = {
    #     "alpha":[i/100.0 for i in range(3000)] + [i for i in range(3, 101)]
    # }
    # scores_method = ["accuracy"]
    # clf = Bayes()
    # clf.load_data(fileName)
    # clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, threshold)
    # clf.param_compare(tuned_parameters, clf.dataset, clf.labels, scores_method)
    # clf.train(clf.dataset, clf.labels)
    # matrix.append(clf.feature_select())

    ## 9. Xgboost
    # tuned_parameters = {
    #     "n_estimators": [i for i in range(200, 500)],
    #     "max_depth": [i for i in range(3, 11)],
    #     "min_child_weight": [i for i in range(10)],
    #     "gamma": [i / 10.0 for i in range(0, 11)],
    #     # "max_features": [i / 20 for i in range(1, 21)]
    # }
    # scores_method = ["accuracy"]
    # clf = Xgboost()
    # clf.load_data(fileName)
    # clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, threshold)
    # clf.param_compare(tuned_parameters, clf.dataset, clf.labels, scores_method)
    # clf.evaluate(clf.dataset, clf.labels)

    # matrix = list(zip(*matrix))
    # FileIO.writeLists(mutation_sort_file, matrix)
