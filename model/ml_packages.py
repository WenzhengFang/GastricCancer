﻿# -*- coding = utf-8 =_=
__author__ = '15624959453@163.com'

import os
from datetime import datetime

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn import svm, linear_model, tree, neighbors, neural_network, ensemble, naive_bayes
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from Dimension_reduction import Feature_selection
import matplotlib.pyplot as plt
import pydotplus

class Parameter(object):
    infoGain_thresh = 0.075  # Threshold of cross entropy for selecting the significant feature
    cross_cv = 3  # Fold of cross validation
    top_gene_counts = 15  # Numbers of top gene for select common genes in second feature selection
    setAside_fold = 10  # The Number of Iteration for set_aside method for evaluating model accuracy
    estimate_method = "set_aside"  # estimate method, belongs to [set_aside, cross_val]
    modelNums_threshold = 2  # To confirm the impact of gene, the number of models appeared gene in common we needed at least.

    SVM_gamma = 0.001  # Kernel coefficient in [rbf,poly,sigmoid], If gamma is ‘auto’ then 1/n_features will be used
    SVM_C = 0.076  # Penalty parameter C of the error term
    SVM_kernel = "linear"  # Specifies the kernel type to be used in the algorithm. one of [linear, poly, rbf, sigmoid]
    SVM_coef = 0.54  # Independent term in kernel function. It is only significant in [poly, sigmoid]
    SVM_degree = 2  # Degree of the polynomial kernel function (poly). Ignored by all other kernels

    RF_n = 22  # The number of trees in the forest
    RF_minSplit = 2  # The minimum number of samples required to split an internal node
    RF_randState = 0  # RandomState instance or None, random_state is the seed used by the random number generator
    RF_maxFeatures = 0.1  # The number of features to consider when looking for the best split, here is a percentage
    RF_criterion = "gini"  # The function to measure the quality of a split, must be one of ["gini", "entropy"]

    GBDT_n = 45  # The number of boosting stages to perform. a large number usually results in better performance
    GBDT_learnRate = 0.1  # learning rate shrinks the contribution of each tree by learning_rate
    GBDT_loss = "exponential"  # loss function to be optimized, one of [exponential, deviance]
    GBDT_maxFeat = 0.05  # The number of features to consider when looking for the best split, float means percentage

    LR_C = 0.14  # Inverse of regularization strength
    LR_penalty = "l2"  # Used to specify the norm used in the penalization
    LR_solver = "liblinear"  # Algorithm to use in the optimization problem
    LR_tol = 0.001  # Tolerance for stopping criteria
    LR_Valid_ratio = 0.33  # The ratio for validation, when divide the dataset into train and validation.
    LR_coef_thresh = 0.2  # threshold used in relation analyse

    KNN_k = 5  # K neighbors to predict the class

    Bayes_alpha = 0.01  # Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)
    Bayes_probThresh = 0.003  # Threshold for feature selection, gene impact lag if min_prob is less than it.

    neural_network_alpha = 0.0  # L2 penalty (regularization term) parameter
    neural_network_hiddenLayer = (4, 7)  # The ith element represents the number of neurons in the ith hidden layer

class PredictModel(object):
    def __init__(self):
        # Feature_selection.__init__(self)
        # self.dataset = dataset
        self.dataset = []
        self.labels = []
        self.titles = []
        self.classifier = None
        self.model_name = ""

    def feature_reduction(self, dataset, titles, labels, thresh):
        """
            Select feature by information gain method.
        """
        feature_oper = Feature_selection()
        new_dataset, new_titles = feature_oper.feature_select(dataset, titles, labels, thresh)
        return new_dataset, new_titles

    def feature_reduction_sec(self, dataset, titles, feat_impace_file):
        """
            Select feature by common gene estimated by several models.
        """
        feature_oper = Feature_selection()
        new_dataset, new_titles = feature_oper.sec_feature_select(
            dataset, titles, feat_impace_file, Parameter.top_gene_counts, Parameter.modelNums_threshold
        )
        return new_dataset, new_titles

    def set_params(self, **kwargs):
        pass

    def load_data(self, mut_analyse_file):
        matrix, labels, titles = [], [], []
        with open(mut_analyse_file, "r") as f1:
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

    def param_tune(self, tuned_parameters, dataset, labels, scores_method):
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
            for params, mean_score, scores in clf.grid_scores_:
                print("Accuracy: %0.3f (+/-%0.03f) for parameter: %r"
                      % (mean_score, scores.std() * 2, params))
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

            self.classifier = clf.best_estimator_

    def train(self, train_set, train_labels):
        self.classifier.fit(train_set, train_labels)

    def predict(self, val_set):
        return self.classifier.predict(val_set)

    def evaluate(self, dataset, labels):
        scores = np.zeros(1)
        if Parameter.estimate_method == "cross_val":
            scores = cross_val_score(self.classifier, dataset, labels, cv=Parameter.cross_cv).reshape(Parameter.cross_cv, )
            print(scores)
            print("Val_Accuracy and Train_Accuracy of model {}: {:.4f}(+/- {:f})".format(
                self.model_name, scores[:].mean(), scores[:].std())
            )
        elif Parameter.estimate_method == "set_aside":
            scores = np.zeros((Parameter.setAside_fold, ))
            train_scores = np.zeros((Parameter.setAside_fold, ))
            for i in range(Parameter.setAside_fold):
                X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=Parameter.LR_Valid_ratio)
                self.train(X_train, Y_train)
                val_acc = self.classifier.score(X_test, Y_test)
                train_acc = self.classifier.score(X_train, Y_train)
                scores[i] = val_acc
                train_scores[i] = train_acc
                print("---> Loop {:d}\n\tVal_Accuracy: {:.4f}\tTrain_Accuracy: {:.4f}".format(i+1, val_acc, train_acc))
            print("Val_Accuracy and Train_Accuracy of model {}:\n\t{:.4f}(+/- {:f})\t{:.4f}(+/- {:f})".format(
                self.model_name, scores[:].mean(), scores[:].std(), train_scores[:].mean(), train_scores[:].std())
            )
        else:
            print("ERROR, p`lease set method in [set_aside, cross_val]")
        return scores

    def visualize(self, **kwargs):
        pass

    def evaluate_roc(self, dataset, labels):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=Parameter.LR_Valid_ratio)
        self.train(X_train, Y_train)
        Y_pred = self.predict(X_test)
        auc_score = roc_auc_score(Y_test, Y_pred)
        # print("AUC on validataion set of model {}: {:.4f}".format(self.model_name, auc_score))
        fpr, tpr, _ = roc_curve(Y_test, Y_pred)
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label=self.model_name)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve with AUC({:.4f})'.format(auc_score.item()))
        plt.legend(loc='best')
        plt.show()
        return auc_score

    def feature_select(self, **kwargs):
        pass

    def save_model(self, mut_analyse_file):
        pass


class LR(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = linear_model.LogisticRegression(
            C=Parameter.LR_C, penalty=Parameter.LR_penalty, tol=Parameter.LR_tol, solver=Parameter.LR_solver
        )
        self.model_name = "LR"

    def set_params(self, C_up, penalty_up):
        self.classifier.set_params(C=C_up, penalty=penalty_up)

    def tumor_analyse(self):
        try:
            clf_coef = self.classifier.coef_
        except:
            raise Exception("warning, classifier are supposed to fit some data.")
        valid_index = np.array(abs(clf_coef) > Parameter.LR_coef_thresh)
        print("Mutation express significantly:\n{}".format("\n".join(self.titles[valid_index])))

    def visualize(self, dataFile, mutation_info):
        y, x = [], []
        plt.figure()
        for i in range(10, 36):
            Parameter.top_gene_counts = i
            self.load_data(dataFile)
            self.dataset, self.titles = self.feature_reduction(self.dataset, self.titles, self.labels, Parameter.infoGain_thresh, mutation_info)
            x.append(self.titles.shape[0])
            score = self.evaluate(self.dataset, self.labels).mean()
            y.append(score)
        plt.plot(x, y)
        plt.title("plot for the number of remain gene and prediction accuracy")
        plt.legend(loc = "upper right")
        plt.xlabel("Number of gene")
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
        self.model_name = "SVM"

    def set_params(self, kernel_string, gamma, c):
        self.classifier.set_params(kernel=kernel_string, gamma=gamma, C=c)

    def visualize(self):
        y, x = [], []
        external = 1.0
        plt.figure()
        self.dataset, self.titles = self.feature_reduction(self.dataset, self.titles, self.labels, Parameter.infoGain_thresh)
        for i in range(1, 1001):
            Parameter.SVM_C = i / external
            x.append(Parameter.SVM_C)
            self.set_params(kernel_string=Parameter.SVM_kernel, gamma=Parameter.SVM_gamma, c=Parameter.SVM_C)
            print("-> Parameter status:\tC({0:f})".format(Parameter.SVM_C), end = "\n\t")
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


class Decision_Tree(PredictModel):

    def __init__(self):
        PredictModel.__init__(self)
        # Scikit-learn uses an small optimised version of the CART algorithm
        self.classifier = tree.DecisionTreeClassifier()
        self.model_name = "Decision_Tree"

    # seems that this function does not make sense.
    def visualize(self, mut_analyse_file):
        dot_data = tree.export_graphviz(self.classifier, out_file=None, feature_names=self.titles, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(mut_analyse_file)
        return


class Bernoulli_Bayes(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = naive_bayes.BernoulliNB(
            alpha=Parameter.Bayes_alpha, class_prior = None
        )
        self.model_name = "Bernoulli_Bayes"

    def feature_select(self):
        log_prob_pos = self.classifier.feature_log_prob_
        real_prob_pos = np.exp(log_prob_pos)
        real_prob_neg = 1 - real_prob_pos
        P_Y1 = float(sum(self.labels)) / len(self.labels)
        P_Y0 = 1 - P_Y1
        infoEnt = -1 * (P_Y1 * np.log(P_Y1) + P_Y0 * np.log(P_Y0))

        ce = []
        for i in range(real_prob_pos.shape[1]):
            P_X1 = np.sum(self.dataset[:, i], axis = 0) / self.dataset.shape[0]
            P_X0 = 1 - P_X1
            ce.append((abs(P_X1 - P_X0), i))

            # P_Y1_X1 = real_prob_pos[1, i] * P_Y1 / P_X1
            # P_Y0_X1 = real_prob_pos[0, i] * P_Y0 / P_X1
            # P_Y1_X0 = real_prob_neg[1, i] * P_Y1 / P_X0
            # P_Y0_X0 = real_prob_neg[0, i] * P_Y0 / P_X0
            #
            # condEnt = (-1*P_X1*(P_Y1_X1*np.log(P_Y1_X1) + P_Y0_X1*np.log(P_Y0_X1)) - P_X0 * (P_Y1_X0*np.log(P_Y1_X0) + P_Y0_X0*np.log(P_Y0_X0)))
            # infoGain = infoEnt - condEnt
            # ce.append((infoGain, i))
        ce.sort(key=lambda x: x[0], reverse=True)
        content = ["BNBayes"] + ["{}({:.6f})".format(self.titles[x[1]], x[0]) for x in ce]

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
        self.model_name = "Xgboost"


class RandomForest(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = ensemble.RandomForestClassifier(
            n_estimators=Parameter.RF_n, max_depth=None, min_samples_split=Parameter.RF_minSplit,
            random_state=Parameter.RF_randState, max_features=Parameter.RF_maxFeatures, criterion=Parameter.RF_criterion
        )
        self.model_name = "RandomForest"

    def feature_select(self):
        feature_impt = self.classifier.feature_importances_
        indices = np.argsort(feature_impt)[::-1]
        # feature_impt.sort(key = lambda x: x[0], reverse=True)
        return ["RandomForest"] + ["{:s}({:.6f})".format(self.titles[indices[f]], feature_impt[indices[f]]) for f in range(feature_impt.shape[0])]


class GBDT(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = ensemble.GradientBoostingClassifier(
            n_estimators=Parameter.GBDT_n, learning_rate=Parameter.GBDT_learnRate,
            max_features=Parameter.GBDT_maxFeat
        )
        self.model_name = "GBDT"

    def feature_select(self):
        feature_impt = [(cf, i) for i, cf in enumerate(self.classifier.feature_importances_)]
        feature_impt.sort(key = lambda x: x[0], reverse=True)
        return ["GBDT"] + ["{:s}({:.6f})".format(self.titles[ind[1]], ind[0]) for ind in feature_impt]


class KNN(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = neighbors.KNeighborsClassifier(
            n_neighbors=Parameter.KNN_k, algorithm="auto"
        )
        self.model_name = "KNN"


class ShallowNetwork(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = neural_network.MLPClassifier(
            solver="lbfgs", alpha=Parameter.neural_network_alpha,
            hidden_layer_sizes=Parameter.neural_network_hiddenLayer, activation="relu"
        )
        self.model_name = "ShallowNetwork"


def model_para_tune_one(mut_analyse_file, threshold, models_tuned_parameters, scores, model_array):
    classifiers = []
    for model_name in model_array:
        clf = eval("{}()".format(model_name))
        clf.load_data(mut_analyse_file)
        clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, threshold)
        clf.param_tune(models_tuned_parameters[model_name], clf.dataset, clf.labels, scores)
        # clf.evaluate(clf.dataset, clf.labels)
        classifiers.append(clf)
    return classifiers

def evaluate_model(classifiers):
    scores = {}
    for clf in classifiers:
        scores[clf.model_name] = clf.evaluate(clf.dataset, clf.labels)
    return scores

def feat_importance_est(classifiers, valid_modelnames, mutation_sort_file):
    feat_matrix = []
    for clf in classifiers:
        if clf.model_name in valid_modelnames:
            clf.train(clf.dataset, clf.labels)
            feat_matrix.append(clf.feature_select())
    feat_matrix = list(zip(*feat_matrix))
    with open(mutation_sort_file, "w") as f1:
        for line in feat_matrix:
            f1.write("\t".join(line) + "\n")
    return feat_matrix

def model_para_tune_two(mut_analyse_file, mutation_impact_file, models_tuned_parameters, scores, model_array):
    classifiers = []
    for model_name in model_array:
        clf = eval("{}()".format(model_name))
        clf.load_data(mut_analyse_file)
        clf.dataset, clf.titles = clf.feature_reduction_sec(clf.dataset, clf.titles, mutation_impact_file)
        clf.param_tune(models_tuned_parameters[model_name], clf.dataset, clf.labels, scores)
        # clf.evaluate(clf.dataset, clf.labels)
        classifiers.append(clf)
    return classifiers

def visualize(x, y, model_array, xlabel, ylabel, outputFile):
    plt.figure()
    for i in range(model_array):
        plt.plot(x, y[i], label = model_array[i])
    plt.title("Accuracy Curve of ml_model")
    plt.legend(loc="upper right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle = ":")
    plt.show()
    plt.savefig(outputFile)
    y = list(zip(*y))
    with open(os.path.splitext(outputFile)[0] + ".txt", "w") as f1:
        for ind in range(len(x)):
            f1.write("\t".join([str(x[ind])] + [str(m) for m in y[ind]]) + "\n")
    return True


def find_common(mutation_sort_file, mutation_sort_file_by_common):
    matrix = []
    with open(mutation_sort_file, "r") as f1:
        for lineno, line in enumerate(f1):
            if lineno >= 1:
                matrix.append([ele.split("(")[0] for ele in line.strip("\n").split("\t")])
    total_feature_nums, match_dict, model_nums = len(matrix), {}, len(matrix[0])
    for common_totals in range(1, total_feature_nums+1):
        common_set = set([line[0] for line in matrix[:common_totals]])
        for j in range(1, model_nums):
            common_set &= set([line[j] for line in matrix[:common_totals]])
        for feature in common_set:
            if feature not in match_dict:
                match_dict[feature] = common_totals
    sort_by_commonNums = sorted([[key, str(value)] for key, value in match_dict.items()], key=lambda x:int(x[1]))
    with open(mutation_sort_file_by_common, "w") as f2:
        f2.write("Feature\tUnique_common_nums\n")
        for line in sort_by_commonNums:
            f2.write("\t".join(line)+"\n")
    return True

__end__ = "yes"

if __name__ == "__main__":
    # print(__doc__)
    mut_analyse_file = "D:\\Project_JY\\gastricCancer\\Data\\mutation_identify\\datasetOfPathology_pos.table"
    threshold = 1.40
    mutation_sort_file = "D:\\Project_JY\\gastricCancer\\Data\\mutation_importance\\mutation_importance_by_ml_pos.txt"
    mutation_sort_file_by_common = "D:\\Project_JY\\gastricCancer\\Data\\mutation_importance\\mutation_importance_by_common_pos.txt"

    # model_array = ["LR", "SVM", "Decision_Tree", "Bernoulli_Bayes", "Xgboost", "RandomForest", "GBDT", "KNN", "ShallowNetwork"]
    model_array = ["LR"]
    models_tuned_parameters = {
        "LR": {"penalty": ["l2", "l1"], "C": [i/1000.0 for i in range(1, 1001)] + [i for i in range(2, 101)]},
        "SVM": {
                "gamma":[i/100.0 for i in range(1, 101)],
                "C": [i/100.0 for i in range(1, 101)] + [i for i in range(2, 101)],
                "kernel": ["linear"],
                "degree": [3],
                "coef0": [i/10.0 for i in range(0, 1)]
            },
        "Decision_Tree": {"min_impurity_split": [i/200 for i in range(1, 101)]},
        "Bernoulli_Bayes": {"alpha":[i/100.0 for i in range(3000)] + [i for i in range(3, 101)]},
        "Xgboost": {
                "learning_rate": [i / 100 for i in range(1, 21)],
                "n_estimators": [i for i in range(300, 400)],
                "max_depth": [i for i in range(3, 11)],
                "gamma": [i / 10.0 for i in range(0, 11)],
                "max_features": [i / 20 for i in range(1, 21)]
            },
        "RandomForest": {
                "max_features":[i/100 for i in range(10, 100)],
                "n_estimators": [i for i in range(20, 100)],
                "oob_score": [True],
                "criterion": ["gini"]
            },
        "GBDT": {
                "loss": ["exponential"],
                "learning_rate": [i/100.0 for i in range(1, 101)],
                "n_estimators": [i for i in range(40, 50)],
                "max_features": [i / 100 for i in range(1, 101)],
            },
        "KNN": {
                "n_neighbors": [i for i in range(1, 21)]
            },
        "ShallowNetwork": {
                "alpha": [i/1000.0 for i in range(1000)],
                "hidden_layer_sizes": [(i, j) for i in range(2, 20) for j in range(2, 20)],
            }
    }
    models_tuned_parameters_assigned = {
        "LR": {"penalty": ["l2"], "C": [0.339]},
        "SVM": {"gamma":[0.01], "C": [0.03], "kernel": ["linear"], "coef0": [0.0]},
        "Decision_Tree": {"max_leaf_nodes": [20], "min_impurity_split": [0.008]},
        "Bernoulli_Bayes": {"alpha": [0.01]},
        "Xgboost": {"learning_rate": [0.09], "n_estimators": [380], "gamma": [0.9]},
        "RandomForest": {"max_features": [0.1], "n_estimators": [20], "oob_score": [True], "criterion": ["gini"]},
        "GBDT": {"loss": ["exponential"], "learning_rate": [0.1], "n_estimators": [45], "max_features": [0.05]},
        "KNN": {"n_neighbors": [12]},
        "ShallowNetwork": {"alpha": [0.0], "hidden_layer_sizes": [(4, 7)]}
    }
    models_tuned_parameters_assigned_sec = {
        "LR": {"penalty": ["l2"], "C": [0.523]},
        "SVM": {"gamma":[0.001], "C": [0.076], "kernel": ["linear"], "coef0": [0.54]},
        "Decision_Tree": {"max_leaf_nodes": [20], "min_impurity_split": [0.008]},
        "Bernoulli_Bayes": {"alpha": [0.01]},
        "Xgboost": {"learning_rate": [0.09], "n_estimators": [380], "gamma": [0.9]},
        "RandomForest": {"max_features": [0.1], "n_estimators": [22], "oob_score": [True], "criterion": ["gini"]},
        "GBDT": {"loss": ["exponential"], "learning_rate": [0.1], "n_estimators": [45], "max_features": [0.05]},
        "KNN": {"n_neighbors": [12]},
        "ShallowNetwork": {"alpha": [0.0], "hidden_layer_sizes": [(4, 7)]}
    }
    scores = ["accuracy"]
    valid_modelnames = {"LR", "SVM", "RandomForest"}


    ## Extra part for debug
    # clf = LR()
    # clf.load_data(mut_analyse_file)
    # sts = clf.titles[np.sum(clf.dataset, axis=0) == 74]
    # find_common(mutation_sort_file, mutation_sort_file_by_common)

    ## First step of feature selection
    # x, y = [], [[] for _ in range(len(model_array))]
    # patterns, pattern = [[1], [i / 1.0 for i in range(2)]], 1
    # for i in patterns[pattern]:
    #     start = datetime.now()
    #
    #     if pattern == 1:
    #         threshold = i
    #     x.append(threshold)
    #     clfs = model_para_tune_one(mut_analyse_file, threshold, models_tuned_parameters, scores, model_array)
    #     # for classifier in clfs:
    #     #     classifier.evaluate_roc(classifier.dataset, classifier.labels)
    #     for j in range(len(clfs)):
    #         y[j].append(evaluate_model(clfs)[model_array[j]].mean())
    #
    #     end = datetime.now()
    #     print('With threshold {0:.2f}, Runtime: {1}s'.format(threshold, str(end - start).split(".")[0]))
    #
    # print(y)
    # if pattern == 1:
    #     xlabel, ylabel, outputImg = "Threshold", "Accuracy", "./parameter_tune_reference/fig1.png"
    #     visualize(x, y, model_array[0], xlabel, ylabel, "")


    ## Second step of feature selection.
    x, y = [], [[] for _ in range(len(model_array))]
    for i in range(36, 37):
        i_start = datetime.now()

        Parameter.top_gene_counts = i
        clfs = model_para_tune_two(mut_analyse_file, mutation_sort_file, models_tuned_parameters, scores, model_array)
        if len(x) > 0 and clfs[0].dataset.shape[1] == x[-1]:
            continue
        feature_nums = clfs[0].dataset.shape[1]
        x.append(feature_nums)
        for j in range(len(clfs)):
            y[j].append(evaluate_model(clfs)[model_array[j]].mean())

        i_end = datetime.now()
        print('Top {0} gene set, {1} genes remained, Runtime: {2}s'.format(i, feature_nums, str(i_end - i_start).split(".")[0]))

    # print(y)
    # if pattern == 1:
    #     xlabel, ylabel, outputImg = "FeatureNums", "Accuracy", "./parameter_tune_reference/fig1.png"
    #     visualize(x, y, model_array[0], xlabel, ylabel, "")

