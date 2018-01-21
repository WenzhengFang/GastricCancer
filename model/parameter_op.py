# -*- coding : utf-8 -*-

"""
* What is this module about?
It consists of all the hybrid parameter used in modules including feature selection and model estimation.

* References:
http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
"""


## Parameter for model estimation
_top_gene_counts = 15  # Numbers of top gene for select common genes in second feature selection
_commonModel_bottom = None  # To confirm the impact of gene, the number of models appeared gene in common we needed at least.

_estimate_method = "set_aside"  # estimate method, belongs to [set_aside, cross_val]
_setAside_iterTimes = 10  # The Number of Iteration for set_aside method for evaluating model accuracy
_setAside_partionRatio = 0.33  # The ratio for validation, when divide the dataset into train and validation.
_crossVal_fold = 3  # Fold of cross validation


## Hyper parameter for model training

# All range of hyperparameters used for grid search of nine model
_hyper_paras = {
    "LR": {"penalty": ["l2", "l1"], "C": [i / 1000.0 for i in range(1, 1001)] + [i for i in range(2, 101)]},
    "SVM": {
        "gamma": [i / 100.0 for i in range(1, 101)],
        "C": [i / 100.0 for i in range(1, 101)] + [i for i in range(2, 101)],
        "kernel": ["linear"],
        "degree": [3],
        "coef0": [i / 10.0 for i in range(0, 1)]
    },
    "Decision_Tree": {"min_impurity_split": [i / 200 for i in range(1, 101)]},
    "Bernoulli_Bayes": {"alpha": [i / 100.0 for i in range(3000)] + [i for i in range(3, 101)]},
    "Xgboost": {
        "learning_rate": [i / 100 for i in range(1, 21)],
        "n_estimators": [i for i in range(300, 400)],
        "max_depth": [i for i in range(3, 11)],
        "gamma": [i / 10.0 for i in range(0, 11)],
        "max_features": [i / 20 for i in range(1, 21)]
    },
    "RandomForest": {
        "max_features": [i / 100 for i in range(10, 100)],
        "n_estimators": [i for i in range(20, 100)],
        "oob_score": [True],
        "criterion": ["gini"]
    },
    "GBDT": {
        "loss": ["exponential"],
        "learning_rate": [i / 100.0 for i in range(1, 101)],
        "n_estimators": [i for i in range(40, 50)],
        "max_features": [i / 100 for i in range(1, 101)],
    },
    "KNN": {
        "n_neighbors": [i for i in range(1, 21)]
    },
    "ShallowNetwork": {
        "alpha": [i / 1000.0 for i in range(1000)],
        "hidden_layer_sizes": [(i, j) for i in range(2, 20) for j in range(2, 20)],
    }
}


# Pattern 1 of stable hyperparameters used for grid search of nine model
_assigned_hyper_paras_1 = {
    "LR": {"penalty": ["l2"], "C": [100]},
    "SVM": {"gamma": [0.01], "C": [0.03], "kernel": ["linear"], "coef0": [0.0]},
    "Decision_Tree": {"max_leaf_nodes": [20], "min_impurity_split": [0.008]},
    "Bernoulli_Bayes": {"alpha": [0.01]},
    "Xgboost": {"learning_rate": [0.09], "n_estimators": [380], "gamma": [0.9]},
    "RandomForest": {"max_features": [0.1], "n_estimators": [20], "oob_score": [True], "criterion": ["gini"]},
    "GBDT": {"loss": ["exponential"], "learning_rate": [0.1], "n_estimators": [45], "max_features": [0.05]},
    "KNN": {"n_neighbors": [12]},
    "ShallowNetwork": {"alpha": [0.0], "hidden_layer_sizes": [(4, 7)]}
}

# Pattern 2 of stable hyperparameters used for grid search of nine model
_assigned_hyper_paras_2 = {
    "LR": {"penalty": ["l2"], "C": [0.523]},
    "SVM": {"gamma": [0.001], "C": [0.076], "kernel": ["linear"], "coef0": [0.54]},
    "Decision_Tree": {"max_leaf_nodes": [20], "min_impurity_split": [0.008]},
    "Bernoulli_Bayes": {"alpha": [0.01]},
    "Xgboost": {"learning_rate": [0.09], "n_estimators": [380], "gamma": [0.9]},
    "RandomForest": {"max_features": [0.1], "n_estimators": [22], "oob_score": [True], "criterion": ["gini"]},
    "GBDT": {"loss": ["exponential"], "learning_rate": [0.1], "n_estimators": [45], "max_features": [0.05]},
    "KNN": {"n_neighbors": [12]},
    "ShallowNetwork": {"alpha": [0.0], "hidden_layer_sizes": [(4, 7)]}
}

# Performance metrics method to evaluate the predictions on the test set
_scores = ["accuracy"]

# Specified model to evaluate the feature importance
_valid_modelnames = {"LR", "SVM", "RandomForest"}