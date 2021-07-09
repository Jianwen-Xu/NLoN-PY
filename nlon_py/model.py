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
    MLPClassifier(alpha=1, max_iter=1000),
    GaussianNB()]

dict_name_classifier = dict(zip(names, classifiers))


def NLoNModel(X, y, nfolds=10, model_name='Nearest Neighbors'):
    X = X[:100]
    y = y[:100]
    if model_name in dict_name_classifier:
        classifier = dict_name_classifier[model_name]
    else:
        classifier = dict_name_classifier['Nearest Neighbors']
    kf = KFold(n_splits=nfolds)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print('%.2f' % score)
    return classifier
