# -*- coding : utf-8 -*-

from ml_packages_Lauren import *
# from ml_packages_WHO import *
import parameter_op as pm

import sys
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.append("../")

########################################################################################################################
# 1.parameter options

# valid_gene_array = ["OR52N1", "TMEM110", "TMEM110-MUSTN1", "AP2B1", "ANKRD12"]
# valid_gene_array = ["COL21A1", "FRG1", "ZNF285", "SLC25A5", "C2orf78", "AHNAK2", "NBPF10", "PRIM2", "PABPC1",
#                     "ANKRD20A1", "ANKRD20A3", "PRG4", "MUC4", "OR52N1", "TMEM110", "TMEM110-MUSTN1", "KRT18",
#                     "MUC3A", "MUC3A", "TCP10", "AP2B1", "PLCE1", "PCDHB6", "MBD1", "DDI2", "KRT28", "ADGRG4",
#                     "POTEF", "ANKRD12", "NBPF1"]
valid_gene_array = (
    "CENPF", 'OR52N1', "SP100", "TMEM177", "OR10G7", "OR4M2", "PSPH", "HNRNPA2B1", "PCDHA3",
    "KIAA2026", 'FOXJ3', "TMEM87B", "ZNF721", "OR4K5", "DDX20", "AFF4", "MTUS2", "CCDC180",
    "TFAM", 'FMN2', 'BCO1', "ANKRD12", "KRTAP10", "SLC25A19", "MTMR6", "TMEM110", "MED13L",
    "CRIM1", 'FAM221A', 'DMRT2', "CASC3", "ZNF578", "TRIM45", "TMEM110-MUSTN1", "FAM21C",
    "SLC34A1", 'DEC1', 'ATG2B', "SMARCE1", "AP2B1"
)
valid_site_array = (
    'chr6_56044814_56044814_C_A', 'chr4_190874234_190874234_C_T', 'chr19_44892153_44892153_G_C',
    'chrX_118605001_118605001_G_T', 'chr2_74043334_74043334_A_C', 'chr14_105417894_105417894_C_G',
    'chr1_145293566_145293566_G_A', 'chr6_57398226_57398226_T_A', 'chr8_101724606_101724606_G_A',
    'chr9_67938633_67938633_G_T', 'chr1_186276565_186276565_A_C', 'chr3_195511609_195511609_A_G',
    'chr11_5809548_5809548_G_A', 'chr3_52886695_52886695_G_T', 'chr12_53343084_53343084_G_C',
    'chr7_100551173_100551173_G_A', 'chr7_100552358_100552358_C_G', 'chr6_167786750_167786750_C_A',
    'chr17_33998774_33998774_T_G', 'chr10_96039597_96039597_G_C', 'chr5_140531291_140531291_A_C',
    'chr18_47800179_47800179_G_C', 'chr1_15953234_15953234_A_G', 'chr17_38955961_38955961_G_A',
    'chrX_135426968_135426968_C_A', 'chr2_130832444_130832444_C_A', 'chr18_9255982_9255982_A_G',
    'chr1_16901021_16901021_A_G '
)
gene_dataset_file_tcga = "D:\\Project_JY\\gastricCancer\\Data\\input_dataset\\TCGA\\dataset_geneFeature_from_tcga.table"
sites_dataset_file_tcga = "D:\\Project_JY\\gastricCancer\\Data\\input_dataset\\TCGA\\dataset_siteFeature_from_tcga" \
                          ".table "
gene_dataset_file_tmu = "D:\\Project_JY\\gastricCancer\\Data\\input_dataset\\gene\\datasetOfPathology_lauren_gene.table"
sites_dataset_file_tmu = "D:\\Project_JY\\gastricCancer\\Data\\input_dataset\\pos\\datasetOfPathology_pos.table"
model_name_1 = "LR"
model_name_2 = "LR"
method, paras, scores, threshold = "thresh", pm._hyper_paras, ["accuracy"], 0.0
flag = "gene"
valid_array = valid_gene_array if flag == "gene" else valid_site_array

########################################################################################################################
# 2.feature compare and information statistic
# clf_1 = eval("{}()".format(model_name_1))
# clf_2 = eval("{}()".format(model_name_2))
# clf_1.load_data(gene_dataset_file_tcga if flag == "gene" else sites_dataset_file_tcga, data_type='tz_who')
# clf_2.load_data(gene_dataset_file_tmu if flag == "gene" else sites_dataset_file_tmu)
# tmu_gene_set = set(clf_2.titles)
# tcga_gene_set = set(clf_1.titles)
# inter_set = tmu_gene_set & tcga_gene_set
# print("The common gene nums: {}".format(len(inter_set)))

########################################################################################################################
# 3.model compare
# clf_1 = eval("{}()".format(model_name_1))
# clf_2 = eval("{}()".format(model_name_2))
# clf_1.load_data(gene_dataset_file_tcga if flag == "gene" else sites_dataset_file_tcga)
# clf_2.load_data(gene_dataset_file_tmu if flag == "gene" else sites_dataset_file_tmu)
# tcga_titles = dict([(clf_1.titles[i], i) for i in range(clf_1.titles.shape[0])])
# tmu_titles = dict([(clf_2.titles[i], i) for i in range(clf_2.titles.shape[0])])
# feature_select_index_tcga = [
#     tcga_titles[valid_array[i]] if valid_array[i] in tcga_titles else -1
#     for i in range(len(valid_array))
# ]
# feature_select_index_tmu = [
#     tmu_titles[valid_array[i]] if valid_array[i] in tmu_titles else -1
#     for i in range(len(valid_array))
# ]
# clf_1.titles = np.array(valid_array)
# clf_1.dataset = np.array([
#         clf_1.dataset[:, ind] if ind > 0 else np.array([0 for i in range(clf_1.dataset.shape[0])])
#         for ind in feature_select_index_tcga
# ]).T
# clf_2.titles = np.array(valid_array)
# clf_2.dataset = clf_2.dataset[:, feature_select_index_tmu]
# clf_2.param_tune(paras["LR"], clf_2.dataset, clf_2.labels, scores)
# clf_2.train(clf_2.dataset, clf_2.labels)
# pred = clf_2.predict(clf_1.dataset)
# print("Accuracy of TCGA: {}\n".format(clf_2.classifier.score(clf_1.dataset, clf_1.labels)))
# print(classification_report(clf_1.labels, pred))



########################################################################################################################
# 4.validation for model selection
rg_1st, rg_2nd, iters = 500, 20, 50
accs = np.zeros((iters, ))
for i in range(iters):
    model_name = 'LR'

    clf = eval("{}()".format(model_name))
    clf.load_data(gene_dataset_file_tmu)

    corr_array = np.array([midd(clf.dataset[:, i], clf.labels) for i in range(clf.dataset.shape[1])])
    corr_sort_ind = np.argsort(np.abs(corr_array))[::-1]
    clf.dataset = clf.dataset[:, corr_sort_ind[:rg_1st]]

    # tmu_titles = dict([(clf.titles[i], i) for i in range(clf.titles.shape[0])])
    # feature_select_index_tmu = [
    #     tmu_titles[valid_array[i]] if valid_array[i] in tmu_titles else -1
    #     for i in range(len(valid_array))
    # ]
    # clf.titles = np.array(valid_array)
    # clf.dataset = clf.dataset[:, feature_select_index_tmu]

    # fold = 3
    # valid_accs = np.zeros((fold, ))
    # kf = StratifiedKFold(n_splits=fold, shuffle=True)
    # for j, (train_index, valid_index) in enumerate(kf.split(clf.dataset, clf.labels)):
    #     # print(clf.labels[train_index])
    #     train_X, train_y = clf.dataset[train_index], [clf.labels[ind] for ind in train_index]
    #     valid_X, valid_y = clf.dataset[valid_index], [clf.labels[ind] for ind in valid_index]
    #
    #     clf.param_tune(paras[model_name], train_X, train_y, ["accuracy"])
    #     clf.train(train_X, train_y)
    #     coef = clf.classifier.coef_[0]
    #     sort_ind = np.argsort(np.abs(coef))[::-1]
    #
    #     filted_train_X = train_X[:, sort_ind[:10]]
    #     filted_valid_X = valid_X[:, sort_ind[:10]]
    #
    #     clf.param_tune(paras[model_name], filted_train_X, train_y, ["accuracy"])
    #     clf.train(filted_train_X, train_y)
    #     valid_accs[j] = clf.classifier.score(filted_valid_X, valid_y)
    #     print("Valid accuracy:{}\n".format(clf.classifier.score(filted_valid_X, valid_y)))
    #     print(classification_report(valid_y, clf.predict(filted_valid_X)))
    # accs[i] = valid_accs.mean()

    clf.dataset, x_valid, clf.labels, y_valid = train_test_split(
        clf.dataset, clf.labels, test_size=0.33, stratify=clf.labels
    )

    clf.param_tune(paras[model_name], clf.dataset, clf.labels, ["accuracy"])
    clf.train(clf.dataset, clf.labels)
    coef = clf.classifier.coef_[0]
    sort_ind = np.argsort(np.abs(coef))[::-1]

    clf.dataset = clf.dataset[:, sort_ind[:rg_2nd]]
    x_valid = x_valid[:, sort_ind[:rg_2nd]]

    clf.param_tune(paras[model_name], clf.dataset, clf.labels, ["accuracy"])
    clf.train(clf.dataset, clf.labels)
    accs[i] = clf.classifier.score(x_valid, y_valid)
    print("Valid accuracy:{}\n".format(clf.classifier.score(x_valid, y_valid)))
    print(classification_report(y_valid, clf.predict(x_valid)))

########################################################################################################################
# 5.
