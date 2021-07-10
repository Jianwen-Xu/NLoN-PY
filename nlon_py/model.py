"""Machine learning models for nlon-py."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = ["Nearest Neighbors", "Linear SVM",
         "Decision Tree", "Random Forest", "Neural Net",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=300),
    GaussianNB()]

dict_name_classifier = dict(zip(names, classifiers))


def NLoNModel(X, y, nfolds=10, model_name='Nearest Neighbors'):
    X = X[:100]
    y = y[:100]
    if model_name in dict_name_classifier:
        clf = dict_name_classifier[model_name]
    else:
        clf = dict_name_classifier['Nearest Neighbors']
    scores = cross_val_score(clf, X, y, cv=nfolds)
    print(f'{scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}')
    return clf


def CompareModels(X, y):
    X = X[:1000]
    y = y[:1000]
    for key, clf in dict_name_classifier.items():
        scores = cross_val_score(clf, X, y, cv=10)
        print(
            f'{key}: {scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}')
