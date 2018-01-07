# -*- coding = utf-8 =_=

__author__ = '15624959453@163.com'

import os
import sys
import Tools.IO as IO

# import graphviz
import numpy as np
from sklearn import datasets
from sklearn import feature_selection
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, LatentDirichletAllocation
import matplotlib.pyplot as plt

class entropy_cal(object):

    def infoEntropy(self, x):
        """
            calculate information entropy H(x)
        """
        x_value_list = set([x[i] for i in range(x.shape[0])])
        ent = 0.0
        for x_value in x_value_list:
            p = float(x[x == x_value].shape[0]) / x.shape[0]
            logp = np.log2(p)
            ent -= p * logp
        return ent

    def condEntropy(self, x, y):
        """
            calculate ent H(y|x)
        """
        x_value_list = set([x[i] for i in range(x.shape[0])])
        ent = 0.0
        for x_value in x_value_list:
            sub_y = y[x == x_value]
            temp_ent = self.infoEntropy(sub_y)
            ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent
        return ent

    def infomationGain(self, x, y):
        y_np = np.array(y)
        base_ent = self.infoEntropy(y_np)
        condition_ent = self.condEntropy(x, y_np)
        ent_grap = base_ent - condition_ent
        return ent_grap

class Feature_selection(object):
    def __init__(self):
        self.dataset = np.zeros((1))
        self.labels = []
        self.titles = []

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

    def corrcoef_calculate(self, dataset, labels):
        N, M = dataset.shape
        content = np.zeros((N, M+1))
        content[:, 0:M] = dataset
        content[:, -1] = labels
        corr_matrix = np.array([np.corrcoef(dataset[:, i], labels)[0, 1] for i in range(dataset.shape[1])])
        # corr_norm = corr_matrix[-1:, :-1].reshape(-1, )
        corr_matrix[np.isnan(corr_matrix)] = 0
        return corr_matrix
        # return corr_matrix

    def infoGain_calculate(self, dataset, labels):
        N, M = dataset.shape
        ent, res = entropy_cal(), []
        for i in range(M):
            x = dataset[:, i]
            res.append(ent.infomationGain(x, labels))
        return np.array(res, dtype=np.float32)

    def output_corr(self, corrcoef_file):
        a = self.corrcoef_calculate(self.dataset, self.labels)
        b = self.infoGain_calculate(self.dataset, self.labels)
        c = np.abs(a) / np.max(np.abs(a), axis=-1) + 1 * b / np.max(b)
        gene_corr = [["gene", "Person_correlation", "Infomation_gain", "Combined_feature_score"]] \
                    + [[self.titles[i], a[i], b[i], c[i]] for i in range(len(self.titles))]

        with open(corrcoef_file, "w") as f1:
            for line in gene_corr:
                f1.write("\t".join(list(map(str, line))) + "\n")
        return

    def feature_select(self, dataset, titles, labels, thresh):
        a = self.corrcoef_calculate(dataset, labels)
        b = self.infoGain_calculate(dataset, labels)
        cb_score = np.abs(a) / np.max(np.abs(a), axis=-1) + 1 * b / np.max(b)

        # infoGainTable = self.infoGain_calculate(dataset, labels)

        feature_select_index = cb_score >= thresh
        feature_select_name = titles[feature_select_index]
        filter_dataset = dataset[:, feature_select_index]
        return filter_dataset, feature_select_name

    def sec_feature_select(self, dataset, titles, feat_impor_file, top_counts=20, common_threshold=None):
        matrix = IO.FileIO.readLists(feat_impor_file)
        if common_threshold == None:
            common_threshold = len(matrix[0])
        matrix_T_clear = list(zip(*[[ele.split("(")[0] for ele in x] for x in matrix[1:top_counts+1]]))
        total_gene_set = dict([[gene, 1] for gene in matrix_T_clear[0]])
        common_set = set()
        for rowByModel in matrix_T_clear[1:]:
            for gene in rowByModel:
                if gene in total_gene_set:
                    total_gene_set[gene] += 1
                else:
                    total_gene_set[gene] = 1
        for gene, counts in total_gene_set.items():
            if counts >= common_threshold:
                common_set.add(gene)
        feature_select_index = [i for i in range(titles.shape[0]) if titles[i] in common_set]
        filter_titles = titles[feature_select_index]
        filter_dataset = dataset[:, feature_select_index]
        # print(filter_titles)
        return filter_dataset, filter_titles


class Dimension_Reduce(object):

    def dimReduction_pca(self, dataset, n_components):
        pca = PCA(n_components = n_components)
        dataset_pca = pca.fit_transform(dataset)
        return dataset_pca

    def dimReduction_ipca(self, dataset, n_components, batch_size):
        ipca = IncrementalPCA(n_components=n_components, batch_size= batch_size)
        dataset_ipca = ipca.fit_transform(dataset)
        return dataset_ipca

    def dimReduction_kpca(self, dataset, kernel, fit_inverse, gamma):
        kpca = KernelPCA(kernel=kernel, fit_inverse_transform=fit_inverse, gamma=gamma)
        dataset_kpca = kpca.fit_transform(dataset)
        dataset_back = kpca.inverse_transform(dataset_kpca)
        return dataset_kpca

    def demo(self):
        n, batch_size = 2, 10
        iris = datasets.load_iris()
        iris_data = iris.data
        colors = ['navy', 'turquoise', 'darkorange']
        X_pca = self.dimReduction_pca(iris_data, n)
        X_ipca = self.dimReduction_ipca(iris_data, n, batch_size)
        y = iris.target

        for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
            plt.figure(figsize=(8, 8))
            for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
                plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                            color=color, lw=2, label=target_name)

            if "Incremental" in title:
                err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
                plt.title(title + " of iris dataset\nMean absolute unsigned error "
                                  "%.6f" % err)
            else:
                plt.title(title + " of iris dataset")
            plt.legend(loc="best", shadow=False, scatterpoints=1)
            plt.axis([-4, 4, -1.5, 1.5])

        plt.show()


if __name__ == "__main__":
    # print(__doc__)
    fileName = "D:\\Project_JY\\gastricCancer\\Data\\mutation_identify\\datasetOfPathology_pos.table"
    corrcoef_file = "D:\\Project_JY\\gastricCancer\\Data\\mutation_identify\\corrcoef_of_muationAndPathology_pos.table"
    mutation_sort_file = "D:\\Project_JY\\gastricCancer\\Data\\mutation_importance\\mutation_importance_by_ml_pos.txt"
    lambd = 1

    fs = Feature_selection()
    fs.load_data(fileName)
    cbScore_threshold = 1.33

    datasets_select, feature_selection = fs.feature_select(fs.dataset, fs.titles, fs.labels, cbScore_threshold)
    # print(feature_selection)
    # gene_corr = [["gene", "Person_correlation", "Infomation_gain"]] + [[fs.titles[i], a[0, i], b[i]] for i in range(len(fs.titles))]
    # fs.output_corr(corrcoef_file)

    # datasets_select, feature_selection = fs.sec_feature_select(fs.dataset, fs.titles, mutation_sort_file, 15, 3)


    # dr = Dimension_Reduce()
    # dr.demo()

