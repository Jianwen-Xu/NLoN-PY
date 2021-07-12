"""Machine learning models for nlon-py."""
<<<<<<< HEAD
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (RandomizedSearchCV, cross_val_score,
                                     train_test_split)
=======
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
>>>>>>> e89b9f91a8c615791562fd339e67b2e83b5e7837
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

<<<<<<< HEAD
from nlon_py.data.make_data import get_category_dict
from nlon_py.features import (Character3GramsForTest, ComputeFeatures,
                              ConvertFeatures, TriGramsAndFeatures)

=======
>>>>>>> e89b9f91a8c615791562fd339e67b2e83b5e7837
names = ["Nearest Neighbors", "Linear SVM",
         "Decision Tree", "Random Forest", "Neural Net",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=300),
    GaussianNB()]

dict_name_classifier = dict(zip(names, classifiers))


def NLoNModel(X, y, features=TriGramsAndFeatures, model_name='Nearest Neighbors'):
    if model_name in dict_name_classifier:
        clf = dict_name_classifier[model_name]
    else:
        clf = dict_name_classifier['Nearest Neighbors']
    X = ConvertFeatures(ComputeFeatures(X, features))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)
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


def SearchParameters_KNN(X, y):
    knn = KNeighborsClassifier(n_neighbors=5)
    k_range = range(1, 31)
    weight_options = ['uniform', 'distance']
    param_dist = dict(n_neighbors=k_range, weights=weight_options)
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy')
    # fit
    rand.fit(X, y)

    print(rand.best_score_)
    print(rand.best_params_)
    print(rand.best_estimator_)


def NLoNPredict(clf, X):
    array_data = ConvertFeatures(ComputeFeatures(X, Character3GramsForTest))
    result = clf.predict(array_data)
    category_dict = get_category_dict()
    result = np.vectorize(category_dict.get)(result)
    return dict(zip(X, result))
