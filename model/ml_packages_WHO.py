# -*- coding : utf-8 -*-

"""
* What is this module about?
It consists of nine module applied to lauren classification and
function packaging the process of parameter tuning, trains and evaluatation.

* References:
http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
"""

import sys
import os
from datetime import datetime
import pandas as pd
import scipy.io as sio
import numpy as np

sys.path.append("../")

import parameter_op as pm
from data_process.Dimension_reduction import Feature_selection
from utility.mutual_information import information_gain
from utility.entropy_estimators import midd

from sklearn import svm, linear_model, tree, neighbors, neural_network, ensemble, naive_bayes
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pydotplus



class PredictModel(object):
    def __init__(self):
        self.entire_data = pd.DataFrame(np.zeros((2, 4)))
        self.char_X = np.zeros((2, 3))
        self.labels_Y = np.zeros((2,))
        self.features = np.zeros((3,))
        self.classifier = None
        self.model_name = ""
        self.sample_weight = None

    def set_params(self, **kwargs):
        arg_dict = kwargs
        self.classifier.set_params(**arg_dict)

    def load_data(self, data_origin, data_type):
        # Load data to memory according to the data_type
        if data_type == 'mat':
            mm_scaler = preprocessing.MinMaxScaler()
            data_mat = sio.loadmat("../Data/origin_Data/{}.mat".format(data_origin))
            self.char_X = mm_scaler.fit_transform(data_mat['X'])
            self.labels_Y = data_mat['Y'].reshape(data_mat['Y'].shape[0], )
            self.features = np.char.add('f', np.arange(self.char_X.shape[1]).astype(str))

        elif data_type == 'tz_lauren':
            self.entire_data = pd.read_csv(data_origin, sep="\t")
            self.entire_data = self.entire_data.dropna(axis=0)
            self.features = np.array([
                feat for feat in self.entire_data.columns
                if feat not in {'patient id', 'who_level', 'diff_level', 'lauren_level'}
                ])
            self.char_X = self.entire_data[self.features].values
            self.labels_Y = self.entire_data['who_level'].values

        elif data_type == "tz_who":
            self.entire_data = pd.read_csv(data_origin, sep="\t")
            self.entire_data = self.entire_data.dropna(axis=0)
            self.features = np.array([
                feat for feat in self.entire_data.columns
                if feat not in {'patient id', 'who_level', 'diff_level', "sample_weight", 'lauren_level'}
                ])
            self.char_X = self.entire_data[self.features].values
            self.labels_Y = self.entire_data['lauren_level'].values
            self.sample_weight = self.entire_data['sample_weight'].values

    def param_tune(self, tuned_parameters, data_x, label_y, sample_weight, scores_methods):
        # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33)
        # train_ind, test_ind = sss.split(data_x, label_y)[0]
        # x_train, x_test = data_x[train_ind], data_x[test_ind]
        # y_train, y_test = label_y[train_ind], label_y[test_ind]
        # weight_train, weight_test = sample_weight[train_ind], sample_weight[test_ind]

        for sm in scores_methods:
            print("# Tuning hyper-parameters for {}\n".format(sm))

            clf = GridSearchCV(
                estimator=self.classifier, param_grid=tuned_parameters, cv=3, scoring=sm
            )
            clf.fit(data_x, label_y, sample_weight=sample_weight)
            print("Grid scores on development set by cross validation:\n")
            for params, mean_score, score_array in clf.grid_scores_:
                print("Accuracy: %0.3f (+/-%0.03f) for parameter: %r"
                      % (mean_score, score_array.std() * 2, params))

            print("\nBest parameters and scores set found on development set by {}:\n".format(sm))
            print("Accuracy of cross validation: {:.3f} for parameter: {}\n".format(clf.best_score_, clf.best_params_))

            # print("Detailed classification report on evaluation set:\n")
            # print(">>>The model is trained on the full development set.")
            # print(">>>The scores are computed on the full evaluation set.\n")
            # y_true, y_pred = y_test, clf.predict(x_test)
            # print(classification_report(y_true, y_pred) + "\n")

            self.classifier = clf.best_estimator_

    def train(self, train_X, train_Y, sample_weight):
        self.classifier.fit(train_X, train_Y, sample_weight)

    def predict(self, valid_X):
        return self.classifier.predict(valid_X)

    def evaluate(self, X, y, weight):
        em, fold, it, pr = pm._estimate_method, pm._crossVal_fold, pm._setAside_iterTimes, pm._setAside_partionRatio
        val_scores, train_scores = np.zeros((fold, )), np.zeros((fold, ))

        skf = StratifiedKFold(n_splits=3, shuffle=True)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            weight_train, weight_test = weight[train_index], weight[test_index]
            self.train(X_train, y_train, weight_train)
            val_scores[i] = self.classifier.score(X_test, y_test, weight_test)
            train_scores[i] = self.classifier.score(X_train, y_train, weight_train)

        print("Val_Accuracy: {}".format(val_scores))
        print(
            "Val_Accuracy and Train_Accuracy of model {}: \n{:.4f}(+/- {:f})\t{:.4f}(+/- {:f})".format(
                self.model_name, val_scores[:].mean(), val_scores[:].std(),
                train_scores[:].mean(), train_scores[:].std())
        )
        return val_scores

    def evaluate_roc(self, dataset, labels):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=pm._setAside_partionRatio)
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

    def featSort_byInfoGain(self, x, y):
        info_array = np.array([information_gain(x[:, i], y) for i in range(x.shape[1])])
        sort_ind = np.argsort(info_array)[::-1]
        info_array = info_array[sort_ind]
        return info_array, sort_ind


class LR(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = linear_model.LogisticRegression(
            C=0.2, penalty="l2", tol=0.001, solver="liblinear", multi_class="ovr"
        )
        self.model_name = "LR"


class SVM(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = svm.SVC(
            gamma=0.01, C=0.03, kernel="linear", coef0=0.54, degree=2, decision_function_shape='ovr'
        )
        self.model_name = "SVM"

    def feature_select(self):
        coef_array = [(cf, i) for i, cf in enumerate(self.classifier.coef_[0])]
        coef_array.sort(key=lambda x: abs(x[0]), reverse=True)
        return ["SVM"] + ["{:s}({:.6f})".format(self.features[ind[1]], ind[0]) for ind in coef_array]


class DecisionTree(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        # Scikit-learn uses an small optimised version of the CART algorithm
        self.classifier = tree.DecisionTreeClassifier()
        self.model_name = "Decision_Tree"

    def visualize(self, mut_analyse_file):
        dot_data = tree.export_graphviz(self.classifier, out_file=None, feature_names=self.features,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(mut_analyse_file)
        return


class BernoulliBayes(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = naive_bayes.BernoulliNB(
            alpha=0.01, class_prior=None
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
            P_X1 = np.sum(self.entire_data[:, i], axis=0) / self.entire_data.shape[0]
            P_X0 = 1 - P_X1
            ce.append((abs(P_X1 - P_X0), i))

        ce.sort(key=lambda x: x[0], reverse=True)
        content = ["BNBayes"] + ["{}({:.6f})".format(self.features[x[1]], x[0]) for x in ce]

        return content


# class Xgboost(PredictModel):
#     '''
#         waiting for that the package installed successfully.
#     '''
#
#     def __init__(self):
#         PredictModel.__init__(self)
#         self.classifier = XGBClassifier(
#             learning_rate=0.1, n_estimators=140, max_depth=5, min_child_weight=1,
#             gamma=0, subsample=0.8, colsample_bytree=0.8, objective="binary:logistic"
#         )
#         self.model_name = "Xgboost"


class RandomForest(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = ensemble.RandomForestClassifier(
            n_estimators=22, max_depth=None, min_samples_split=2,
            random_state=0, max_features=0.1, criterion="gini"
        )
        self.model_name = "RandomForest"

    def feature_select(self):
        feature_impt = self.classifier.feature_importances_
        indices = np.argsort(feature_impt)[::-1]
        # feature_impt.sort(key = lambda x: x[0], reverse=True)
        return ["RandomForest"] + ["{:s}({:.6f})".format(self.features[indices[f]], feature_impt[indices[f]]) for f in
                                   range(feature_impt.shape[0])]


class GBDT(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = ensemble.GradientBoostingClassifier(
            n_estimators=45, learning_rate=0.1,
            max_features=45, loss="exponential"
        )
        self.model_name = "GBDT"

    def feature_select(self):
        feature_impt = [(cf, i) for i, cf in enumerate(self.classifier.feature_importances_)]
        feature_impt.sort(key=lambda x: x[0], reverse=True)
        return ["GBDT"] + ["{:s}({:.6f})".format(self.features[ind[1]], ind[0]) for ind in feature_impt]


class KNN(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = neighbors.KNeighborsClassifier(
            n_neighbors=12, algorithm="auto"
        )
        self.model_name = "KNN"


class ShallowNetwork(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = neural_network.MLPClassifier(
            solver="lbfgs", alpha=0.0,
            hidden_layer_sizes=(4, 7), activation="relu"
        )
        self.model_name = "ShallowNetwork"


def model_para_tune(mut_analyse_file, models_tuned_parameters, scores, model_array, method, **kwargs):
    classifiers = []
    for model_name in model_array:
        clf = eval("{}()".format(model_name))
        clf.load_data(mut_analyse_file)
        if method == "thresh":
            clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, kwargs["threshold"])
        elif method == "count":
            clf.dataset, clf.titles = clf.feature_reduction_sec(
                clf.dataset, clf.titles, kwargs["feature_sort"], kwargs["topNums"], kwargs["commonModel_bottom"]
            )
        else:
            print("ERROR!!!")
        clf.param_tune(models_tuned_parameters[model_name], clf.dataset, clf.labels, scores)
        classifiers.append(clf)
    return classifiers


def evaluate_model(classifiers):
    scores = []
    for clf in classifiers:
        scores.append(clf.evaluate(clf.dataset, clf.labels))
    return scores


def chara_im_assess_by_ml(classifiers, valid_modelnames, mutation_sort_file):
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


def visualize(x, y, model_array, xlabel, ylabel, outputFile):
    plt.figure()
    for i in range(len(model_array)):
        plt.plot(x, y[i], label=model_array[i])
    plt.title("Accuracy Curve of ml_model")
    plt.legend(loc="upper right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle=":")
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
    for common_totals in range(1, total_feature_nums + 1):
        common_set = set([line[0] for line in matrix[:common_totals]])
        for j in range(1, model_nums):
            common_set &= set([line[j] for line in matrix[:common_totals]])
        for feature in common_set:
            if feature not in match_dict:
                match_dict[feature] = common_totals
    sort_by_commonNums = sorted([[key, str(value)] for key, value in match_dict.items()], key=lambda x: int(x[1]))
    with open(mutation_sort_file_by_common, "w") as f2:
        f2.write("Feature\tUnique_common_nums\n")
        for line in sort_by_commonNums:
            f2.write("\t".join(line) + "\n")
    return True


def estimate_roughly(model_array, mut_dataset_file, directory):
    # Determine the threshold range of correlation scores.
    if not os.path.exists(directory):
        os.mkdir(directory)
    feat_screen_by_corr, scores = os.path.join(directory, "feat_screen_by_correlation.table"), []
    featOp = Feature_selection()
    featOp.load_data(mut_dataset_file)
    featOp.output_corr(feat_screen_by_corr)
    with open(feat_screen_by_corr, "r") as f1:
        for lineno, line in enumerate(f1):
            if lineno >= 1:
                scores.append(float(line.strip().split("\t")[3]))
    max_score, min_score = max(scores), min(scores)

    paras, scores, valid_models = pm._hyper_paras, pm._scores, pm._valid_modelnames
    feat_screen_by_model = os.path.join(directory, "feat_screen_by_model.table")
    x, y = [], [[] for _ in range(len(model_array))]
    i = int(min_score * 100)
    while i < int(max_score * 100):
        st = datetime.now()

        score_thresh, x = i / 100.0, x + [i / 100.0]
        clfs = model_para_tune(mut_dataset_file, paras, scores, model_array, "thresh", threshold=score_thresh)
        performs = evaluate_model(clfs)
        for j in range(len(model_array)):
            y[j].append(performs[j].mean())

        i += 1
        ed = datetime.now()
        print('With threshold {0:.4f}, Runtime: {1}s'.format(score_thresh, str(ed - st).split(".")[0]))

    # Make use of the last maximum location to evaluate character power
    ind = 0
    for i in range(len(y[0])):
        if y[0][i] > y[0][ind]:
            ind = i
    score_thresh = x[ind]
    clfs = model_para_tune(mut_dataset_file, paras, scores, model_array, "thresh", threshold=score_thresh)
    chara_im_assess_by_ml(clfs, valid_models, feat_screen_by_model)

    xlabel, ylabel, outputImg = "Threshold of Combined Score", "Accuracy", os.path.join(directory,
                                                                                        "accuracy_trend_by_thresh.png")
    visualize(x, y, model_array, xlabel, ylabel, outputImg)
    return feat_screen_by_model, outputImg


def estimate_meticulously(model_array, mut_dataset_file, feat_screen_by_model, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(feat_screen_by_model, "r") as f1:
        upper = len(f1.readlines()) - 1

    paras, scores, common_models = pm._assigned_hyper_paras_1, pm._scores, pm._commonModel_bottom
    x, y = [], [[] for _ in range(len(model_array))]
    for i in range(1, upper + 1):
        st = datetime.now()

        topNums, flag = i, True
        meta_model = PredictModel()
        meta_model.load_data(mut_dataset_file)
        meta_model.entire_data, meta_model.features = meta_model.feature_reduction_sec(
            meta_model.entire_data, meta_model.features, feat_screen_by_model, topNums, common_models
        )
        feature_nums = meta_model.entire_data.shape[1]
        if feature_nums <= 1 or (len(x) > 0 and feature_nums == x[-1]):
            continue
        else:
            clfs = model_para_tune(
                mut_dataset_file, paras, scores, model_array, "count", feature_sort=feat_screen_by_model,
                topNums=topNums, commonModel_bottom=common_models
            )
            performs = evaluate_model(clfs)
            x.append(feature_nums)
            for j in range(len(model_array)):
                y[j].append(performs[j].mean())

        ed = datetime.now()
        print('Top {0} gene set, {1} genes remained. all models passed, Runtime: {2}s'.format(
            i, feature_nums, str(ed - st).split(".")[0])
        )

    xlabel, ylabel, outputImg = "Num of Remained Features", "Accuracy", os.path.join(directory,
                                                                                     "accuracy_trend_by_count.png")
    feat_sort_by_model = os.path.join(directory, "feature_sort_by_model.table")
    visualize(x, y, model_array, xlabel, ylabel, outputImg)
    find_common(feat_screen_by_model, feat_sort_by_model)
    return feat_sort_by_model


__end__ = "yes"

if __name__ == "__main__":
    # print(__doc__)
    # mut_dataset_file = "D:\\Project_JY\\gastricCancer\\Data\\input_dataset\\pogs\\datasetOfPathology_pos.table"
    # output_dir_pos = "D:\\Project_JY\\gastricCancer\\Result\\who\\pos"

    mut_analyse_file = "D:\\Project_JY\\gastricCancer\\Data\\input_dataset\\gene\\datasetOfPathology_who_gene.table"
    output_dir_gene = "D:\\Project_JY\\gastricCancer\\Result\\who\\gene"
    model_array = ["RandomForest"]
    paras, scores, methods = pm._hyper_paras, ["accuracy"], "thresh"
    bottom, n_feat, iters = 0, 200, 10

    # Extra part for debug
    clf = eval("{}()".format(model_array[0]))
    clf.load_data(mut_analyse_file, 'tz_who')

    # sorted_infoGain, filted_ind = clf.featSort_byInfoGain(clf.char_X, clf.labels_Y)
    # x_selected = clf.char_X[:, filted_ind[:n_feat]]
    # feat_selected = clf.features[filted_ind[:n_feat]]
    # clf.evaluate(x_selected, clf.labels_Y, clf.sample_weight)
    # clf.param_tune(paras[model_array[0]], x_selected, clf.labels_Y, clf.sample_weight, scores)
