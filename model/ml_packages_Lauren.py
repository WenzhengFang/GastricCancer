# -*- coding : utf-8 -*-

"""
* What is this module about?
It consists of nine module applied to lauren classification and
function packaging the process of parameter tuning, trains and evaluatation.

* References:
http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
"""

from data_process.Dimension_reduction import Feature_selection

import sys
import parameter_op as pm
sys.path.append(".")
import os
from datetime import datetime
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from utility.entropy_estimators import midd

import numpy as np
# from xgboost.sklearn import XGBClassifier
from sklearn import svm, linear_model, tree, neighbors, neural_network, ensemble, naive_bayes
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
# import pydotplus


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

    def feature_reduction_sec(self, dataset, titles, feat_impace_file, topNums, commonModel_bottom):
        """
            Select feature by common gene estimated by several models.
        """
        feature_oper = Feature_selection()
        new_dataset, new_titles = feature_oper.sec_feature_select(
            dataset, titles, feat_impace_file, topNums, commonModel_bottom
        )
        return new_dataset, new_titles

    def set_params(self, kwargs):
        self.classifier.set_params(**kwargs)

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
                        matrix.append([int(x) for x in data_array[1:-4]])
                else:
                    titles = data_array[1:-4]
        self.dataset = np.array(matrix)
        self.titles = np.array(titles)
        self.labels = labels

    def param_tune(self, tuned_parameters, dataset, labels, scores_method):
        X_train, X_test, y_train, y_test = train_test_split(
            dataset, labels, test_size=pm._setAside_partionRatio, stratify=labels
        )
        for score in scores_method:
            print("# Tuning hyper-parameters for {}\n".format(score))

            clf = GridSearchCV(
                estimator=self.classifier, param_grid=tuned_parameters, cv=pm._crossVal_fold, scoring=score
            )
            clf.fit(dataset, labels)
            print("Grid scores on development set:\n")
            for params, mean_score, scores in clf.grid_scores_:
                print("Accuracy: %0.3f (+/-%0.03f) for parameter: %r"
                      % (mean_score, scores.std() * 2, params))

            print("\nBest parameters and scores set found on development set by {}:\n".format(score))
            print("--->Accuracy: {:.3f}, parameter: {}\n".format(clf.best_score_, clf.best_params_))

            print("Detailed classification report:\n")
            print(">The model is trained on the full development set.")
            print(">The scores are computed on the full evaluation set.\n")
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
        em, fold, it, pr = pm._estimate_method, pm._crossVal_fold, pm._setAside_iterTimes, pm._setAside_partionRatio
        if em == "cross_val":
            scores = cross_val_score(self.classifier, dataset, labels, cv=fold).reshape(fold, )
            print(scores)
            print("Val_Accuracy and Train_Accuracy of model {}: {:.4f}(+/- {:f})".format(
                self.model_name, scores[:].mean(), scores[:].std())
            )
        elif em == "set_aside":
            scores = np.zeros((it,))
            train_scores = np.zeros((it,))
            for i in range(it):
                X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size=pr)
                self.train(X_train, Y_train)
                val_acc = self.classifier.score(X_test, Y_test)
                train_acc = self.classifier.score(X_train, Y_train)
                scores[i] = val_acc
                train_scores[i] = train_acc
                print(
                    "---> Loop {:d}\n\tVal_Accuracy: {:.4f}\tTrain_Accuracy: {:.4f}".format(i + 1, val_acc, train_acc))
            print("Val_Accuracy and Train_Accuracy of model {}:\n\t{:.4f}(+/- {:f})\t{:.4f}(+/- {:f})".format(
                self.model_name, scores[:].mean(), scores[:].std(), train_scores[:].mean(), train_scores[:].std())
            )
        else:
            print("ERROR, p`lease set method in [set_aside, cross_val]")
        return scores

    def visualize(self, **kwargs):
        pass

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

    def feature_select(self, **kwargs):
        pass

    def save_model(self, mut_analyse_file):
        pass


class LR(PredictModel):
    def __init__(self, **kwargs):
        PredictModel.__init__(self)
        self.classifier = linear_model.LogisticRegression(
            C=0.001, penalty="l2", tol=0.001, solver="liblinear"
        )
        self.model_name = "LR"

    def feature_select(self):
        coef_array = [(cf, i) for i, cf in enumerate(self.classifier.coef_[0])]
        coef_array.sort(key=lambda x: abs(x[0]), reverse=True)
        return ["LR"] + ["{:s}({:.6f})".format(self.titles[ind[1]], ind[0]) for ind in coef_array]


class SVM(PredictModel):
    def __init__(self):
        PredictModel.__init__(self)
        self.classifier = svm.SVC(
            gamma=0.01, C=0.907, kernel="linear", coef0=0.54, degree=2,
        )
        self.model_name = "SVM"

    # def visualize(self):
    #     y, x = [], []
    #     external = 1.0
    #     plt.figure()
    #     self.dataset, self.titles = self.feature_reduction(self.dataset, self.titles, self.labels, Parameter.infoGain_thresh)
    #     for i in range(1, 1001):
    #         Parameter.SVM_C = i / external
    #         x.append(Parameter.SVM_C)
    #         self.set_params(kernel_string=Parameter.SVM_kernel, gamma=Parameter.SVM_gamma, c=Parameter.SVM_C)
    #         print("-> Parameter status:\tC({0:f})".format(Parameter.SVM_C), end = "\n\t")
    #         score = self.evaluate(self.dataset, self.labels).mean()
    #         y.append(score)
    #     plt.plot(x, y)
    #     plt.title("plot for the relationship of C and prediction accuracy in linear kernel")
    #     plt.legend(loc = "upper right")
    #     plt.xlabel("C/{:f}".format(1/external))
    #     plt.ylabel("Accuracy/1")
    #     plt.grid(x)
    #     plt.show()

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
        dot_data = tree.export_graphviz(self.classifier, out_file=None, feature_names=self.titles,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(mut_analyse_file)
        return


class Bernoulli_Bayes(PredictModel):
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
            P_X1 = np.sum(self.dataset[:, i], axis=0) / self.dataset.shape[0]
            P_X0 = 1 - P_X1
            ce.append((abs(P_X1 - P_X0), i))

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
            learning_rate=0.1, n_estimators=140, max_depth=5, min_child_weight=1,
            gamma=0, subsample=0.8, colsample_bytree=0.8, objective="binary:logistic"
        )
        self.model_name = "Xgboost"


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
        return ["RandomForest"] + ["{:s}({:.6f})".format(self.titles[indices[f]], feature_impt[indices[f]]) for f in
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
        return ["GBDT"] + ["{:s}({:.6f})".format(self.titles[ind[1]], ind[0]) for ind in feature_impt]


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
        meta_model.dataset, meta_model.titles = meta_model.feature_reduction_sec(
            meta_model.dataset, meta_model.titles, feat_screen_by_model, topNums, common_models
        )
        feature_nums = meta_model.dataset.shape[1]
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
    # parameter of mutation sites
    # mut_dataset_file = "D:\\Project_JY\\gastricCancer\\Data\\mutation_identify\\datasetOfPathology_pos.table"
    # threshold = 1.40
    # output_dir = "D:\\Project_JY\\gastricCancer\\Result\\pos"

    # parameter of gene
    mut_dataset_file = "D:\\Project_JY\\gastricCancer\\Data\\input_dataset\\gene\\datasetOfPathology_lauren_gene_1.table"
    threshold = 0.2
    output_dir = "D:\\Project_JY\\gastricCancer\\Result\\lauren\\gene"
    train_dataset_file = "D:\\Project_JY\\gastricCancer\\Data\\input_dataset\\gene\\dataset_geneFeat_train.table"
    valid_dataset_file = "D:\\Project_JY\\gastricCancer\\Data\\input_dataset\\gene\\dataset_geneFeat_valid.table"

    # parameter of tcga gene
    # mut_dataset_file = "D:\\Project_JY\\gastricCancer\\Data\\input_dataset\\TCGA\\dataset_geneFeat_from_tcga.table"
    # threshold = 1.40
    # output_dir = "D:\\Project_JY\\gastricCancer\\Result\\lauren\\tcga_gene"

    # model_array = ["LR", "SVM", "Decision_Tree", "Bernoulli_Bayes", "Xgboost", "RandomForest", "GBDT", "KNN",
    #     "ShallowNetwork"]
    model_name = "LR"
    model_array, paras, scores = [model_name], pm._hyper_paras, pm._scores
    clf = eval("{}()".format(model_name))
    clf.load_data(mut_dataset_file)


    # clf_l = eval("{}()".format(model_name))
    # clf_l.load_data(valid_dataset_file)
    # clf.dataset, clf.titles = clf.feature_reduction(clf.dataset, clf.titles, clf.labels, threshold)
    # titles_dict = dict([(clf_l.titles[i], i) for i in range(clf_l.titles.shape[0])])
    # feature_select_index = [
    #     titles_dict[clf.titles[i]] for i in range(len(clf.titles))
    # ]
    # clf_l.dataset, clf_l.titles = clf_l.dataset[:, feature_select_index], clf.titles
    #
    # clf.param_tune(paras[model_name], clf.dataset, clf.labels, scores)
    # # clf.set_params({"C":0.018, "penalty":"l2"})
    # clf.train(clf.dataset, clf.labels)
    # print("Train Accuracy: {}".format(clf.classifier.score(clf.dataset, clf.labels)))
    # print("Valid Accuracy: {}".format(clf.classifier.score(clf_l.dataset, clf_l.labels)))

    # feat_screen_by_model, img1 = estimate_roughly(model_array, train_dataset_file, output_dir)
    # estimate_meticulously(model_array, mut_dataset_file, feat_screen_by_model, output_dir_pos)
