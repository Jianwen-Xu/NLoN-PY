"""Machine learning models for nlon-py."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import (RandomizedSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from nlon_py.data.make_data import get_category_dict
from nlon_py.features import (ComputeFeatures, ConvertFeatures,
                              TriGramsAndFeatures, TriGramsAndFeaturesForTest)

names = ["Nearest Neighbors", "Linear SVM",
         "Decision Tree", "Random Forest", "Neural Net",
         "Naive Bayes", "Glmnet"]

classifiers = [
    KNeighborsClassifier(),
    LinearSVC(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=300),
    GaussianNB(),
    ElasticNetCV(l1_ratio=1, cv=10)]

dict_name_classifier = dict(zip(names, classifiers))


def NLoNModel(X, y, features=TriGramsAndFeatures, model_name='Naive Bayes'):
    if model_name in dict_name_classifier:
        clf = dict_name_classifier[model_name]
    else:
        clf = dict_name_classifier['Naive Bayes']
    X = ConvertFeatures(ComputeFeatures(X, features))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f'{model_name}: {score:.2f} accuracy')
    return clf


def CompareModels(X, y):
    X = X[:100]
    y = y[:100]
    for key, clf in dict_name_classifier.items():
        scores = cross_val_score(clf, X, y, cv=10)
        print(
            f'{key}: {scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}')


def ValidateModel(model, X, y):
    score = cross_val_score(model, X, y, cv=10, scoring='balanced_accuracy')
    print(
        f'10-Fold Cross Validation: {score.mean():.2f} average accuracy with a standard deviation of {score.std():.2f}')


def NLoNPredict(clf, X):
    array_data = ConvertFeatures(
        ComputeFeatures(X, TriGramsAndFeaturesForTest))
    result = clf.predict(array_data)
    category_dict = get_category_dict()
    result = np.vectorize(category_dict.get)(result)
    return dict(zip(X, result))
