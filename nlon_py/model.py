"""Machine learning models for nlon-py."""
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import auc, plot_roc_curve
from sklearn.model_selection import (HalvingGridSearchCV,
                                     StratifiedShuffleSplit, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from nlon_py.data.make_data import get_category_dict
from nlon_py.features import ComputeFeatures, ConvertFeatures

names = ["Nearest Neighbors", "SVM", "Naive Bayes"]

classifiers = [KNeighborsClassifier(),
               SVC(kernel='linear', probability=True, random_state=0),
               ComplementNB()]

dict_name_classifier = dict(zip(names, classifiers))


def NLoNModel(X, y, features=None, model_name='Naive Bayes'):
    if model_name in dict_name_classifier:
        clf = dict_name_classifier[model_name]
    else:
        raise RuntimeError('Param model_name should be in ' + names.__str__())
    clf = make_pipeline(StandardScaler(), clf)
    if features is not None:
        X = ConvertFeatures(ComputeFeatures(X, features))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f'{model_name}: {score:.2f} accuracy')
    return clf


def SearchParams_SVM(X, y):
    C_range = np.logspace(-5, 2, 8)
    C_weight = [None, 'balanced']
    param_grid = dict(C=C_range, class_weight=C_weight)
    clf = dict_name_classifier['SVM']
    search = HalvingGridSearchCV(clf, param_grid, cv=2, random_state=0)
    search.fit(X, y)
    print(search.best_params_)


def SearchParams_KFold(X, y):
    clf = SVC(kernel='linear', probability=True, random_state=0)
    skf = StratifiedShuffleSplit(n_splits=2, random_state=0)
    C_range = [1, 10, 100]
    C_weight = [None, 'balanced']
    param_scores = []
    t_ = time()
    for C_param in C_range:
        clf.C = C_param
        t0 = time()
        print(f'Try C_param:{C_param} start...')
        fold_index = 0
        for train_index, test_index in skf.split(X, y):
            fold_index += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(f'Fold:{fold_index} start...')
            clf.fit(X_train, y_train)
            print(f'Fold:{fold_index} train in {(time() - t0):0.3f}s')
            test_score = clf.score(X_test, y_test)
            print(f'Fold:{fold_index} score in {(time() - t0):0.3f}s')
            print(f'Score:{test_score:.2f}')
            param_scores.append([C_param, test_score])
        print(f'Try C_param:{C_param} done in {(time() - t_):0.3f}s')
    return param_scores


def CompareModels(X, y):
    X = X[:100]
    y = y[:100]
    for key, clf in dict_name_classifier.items():
        scores = cross_val_score(clf, X, y, cv=10)
        print(
            f'{key}: {scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}')


def ValidateModel(model, X, y):
    score = cross_val_score(
        model, X, y, cv=10, scoring='accuracy', error_score=0)
    print(
        f'10-Fold Cross Validation: {score.mean():.2f} average accuracy with a standard deviation of {score.std():.2f}')


def NLoNPredict(clf, X, features=None):
    array_data = X
    if features is not None:
        array_data = ConvertFeatures(ComputeFeatures(X, features))
    result = clf.predict(array_data)
    category_dict = get_category_dict()
    result = np.vectorize(category_dict.get)(result)
    return list(result)
