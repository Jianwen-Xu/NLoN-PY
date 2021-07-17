import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import Normalizer, StandardScaler

from nlon_py.data.make_data import get_category_dict, loadDataFromFiles
from nlon_py.features import (Character3Grams, ComputeFeatures,
                              ConvertFeatures, FeatureExtraction,
                              TriGramsAndFeatures, TriGramsAndFeaturesForTest)
from nlon_py.model import (CompareModels, NLoNModel, NLoNPredict,
                           SearchParams_KFold, SearchParams_SVM, ValidateModel)

pwd_path = os.path.abspath(os.path.dirname(__file__))
modelfile = os.path.join(pwd_path, 'default_model.joblib')
datafile = os.path.join(pwd_path, 'default_data.joblib')
test_corpus = ['This is natural language.',
               'public void NotNaturalLanguageFunction(int i, String s)',
               '''Exception in thread "main" java.lang.NullPointerException
                at com.example.myproject.Book.getTitle(Book.java:16)
                at com.example.myproject.Author.getBookTitles(Author.java:25)
                at com.example.myproject.Bootstrap.main(Bootstrap.java:14)''',
               '''2012-02-02 12:47:03,309 ERROR [com.api.bg.sample] - Exception is 
                :::java.lang.IndexOutOfBoundsException: Index: 0, Size: 0''',
               '''However, I only get the file name, not the file content. When I add enctype=
                "multipart/form-data" to the <form>, then request.getParameter() returns null.''',
               '''The format is the same as getStacktrace, for e.g. 
                I/System.out(4844): java.lang.NullPointerException
                at com.temp.ttscancel.MainActivity.onCreate(MainActivity.java:43)
                at android.app.Activity.performCreate(Activity.java:5248)
                at android.app.Instrumentation.callActivityOnCreate(Instrumentation.java:1110)''',
               ''' Why does my JavaScript code receive a “No 'Access-Control-Allow-Origin' header 
                is present on the requested resource” error, while Postman does not?''']


def buildDefaultModel():
    X, y = loadDefaultData()
    print("[buildDefaultModel] building...")
    t0 = time()
    clf = NLoNModel(X, y, model_name='SVM')
    dump(clf, modelfile)
    print(f"[buildDefaultModel] done in {(time() - t0):0.3f}s")


def loadDefaultModel():
    return load(modelfile)


def testDefaultModel():
    model = loadDefaultModel()
    print(NLoNPredict(model, transform_data(test_corpus)))


def validDefaultModel():
    print("[validDefaultModel] start...")
    t0 = time()
    X, y = loadDefaultData()
    print(f"[validDefaultModel] load data in {(time() - t0):0.3f}s")
    model = loadDefaultModel()
    print(f"[validDefaultModel] load model in {(time() - t0):0.3f}s")
    ValidateModel(model, X, y)
    print(f"[validDefaultModel] done in {(time() - t0):0.3f}s")


def searchParams():
    print("[searchParams] start...")
    t0 = time()
    X, y = loadDefaultData()
    X = Normalizer().fit_transform(X)
    X = X[::2]
    y = y[::2]
    # print(np.where(y == 6))
    SearchParams_KFold(X, y)
    print(f"[searchParams] done in {(time() - t0):0.3f}s")


def compareDifModels():
    X, y = loadDefaultData()
    CompareModels(X, y)


def plotDistribution():
    X, y = loadDefaultData()
    class_dict = get_category_dict()
    unique, counts = np.unique(y, return_counts=True)
    labels = [class_dict[x] for x in unique]
    plt.bar(labels, counts, width=0.5)
    for i, v in enumerate(counts):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.title('Categories Distribution')
    plt.savefig('Distribution.png')


def transform_data(X):
    return ConvertFeatures(ComputeFeatures(X, TriGramsAndFeaturesForTest))


def buildDefaultData():
    print("[buildDefaultData] building...")
    t0 = time()
    X, y = loadDataFromFiles()
    X = transform_data(X)
    dump(dict(data=X, target=y), datafile)
    print(f"[buildDefaultData] done in {(time() - t0):0.3f}s")


def loadDefaultData():
    print("[loadDefaultData] loading...")
    t0 = time()
    data_dict = load(datafile)
    X = data_dict['data']
    y = data_dict['target']
    print(f"[loadDefaultData] done in {(time() - t0):0.3f}s")
    return X, y
