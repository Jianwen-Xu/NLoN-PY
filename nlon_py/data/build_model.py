import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from pandas.core import groupby
from sklearn.model_selection import train_test_split

from nlon_py.data.make_data import get_category_dict, loadDataFromFiles
from nlon_py.data.original_data.make_ori_data import loadOriDataFromFiles
from nlon_py.data.extend_data.make_ext_data import loadExtDataFromFiles
from nlon_py.features import NLoNFeatures
from nlon_py.model import (CompareModels, NLoNModel, NLoNPredict,
                           SearchParams_SVM, ValidateModel,
                           plot_multiclass_roc, plot_twoclass_roc,plot_confusion_matrix)

pwd_path = os.path.abspath(os.path.dirname(__file__))
modelfile = os.path.join(pwd_path, 'default_model.joblib')
datafile = os.path.join(pwd_path, 'default_data.joblib')
ori_datafile = os.path.join(pwd_path, 'original_data.joblib')
ori_modelfile = os.path.join(pwd_path, 'original_model.joblib')
ext_datafile = os.path.join(pwd_path, 'extend_data.joblib')
ext_modelfile = os.path.join(pwd_path, 'extend_model.joblib')
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


def buildDefaultModel(n_classes=7, features='C3_FE', stand=True, kbest=True):
    X, y = loadDefaultData(n_classes=n_classes)
    print("[buildDefaultModel] building...")
    t0 = time()
    clf = NLoNModel(X, y, features, model_name='SVM', stand=stand, kbest=kbest, n_classes=n_classes)
    dump(clf, modelfile, compress='zlib')
    print(f"[buildDefaultModel] done in {(time() - t0):0.3f}s")


def loadDefaultModel():
    return load(modelfile)


def testDefaultModel():
    model = loadDefaultModel()
    print(NLoNPredict(model, test_corpus))


def validDefaultModel():
    print("[validDefaultModel] start...")
    t0 = time()
    X, y = loadDefaultData(n_classes=5)
    print(f"[validDefaultModel] load data in {(time() - t0):0.3f}s")
    model = loadDefaultModel()
    print(f"[validDefaultModel] load model in {(time() - t0):0.3f}s")
    ValidateModel(model, X, y)
    print(f"[validDefaultModel] done in {(time() - t0):0.3f}s")


def searchParams(n_classes=7):
    print("[searchParams] start...")
    t0 = time()
    # X, y = loadDefaultData()
    X, y = loadDefaultData(n_classes=n_classes)
    SearchParams_SVM(X, y)
    print(f"[searchParams] done in {(time() - t0):0.3f}s")


def compareDifModels(n_classes, cv=None):
    print("[compareDifModels] start...")
    t0 = time()
    X, y = loadDefaultData(n_classes)
    CompareModels(X, y, cv=cv)
    print(f"[compareDifModels] done in {(time() - t0):0.3f}s")


def plotDistribution(n_classes=7):
    X, y = loadDefaultData(n_classes)
    class_dict = get_category_dict()
    unique, counts = np.unique(y, return_counts=True)
    labels = [class_dict[x] for x in unique]
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.bar(labels, counts, width=0.5)
    for i, v in enumerate(counts):
        plt.text(i+0.2, v+100, str(v), ha='center', va='bottom')
    plt.ylim(0, 8000)
    plt.ylabel('Samples')
    ax.plot(labels, counts, '--')
    plt.title('Categories Distribution and AUC performance')

    # auc = np.array([0.91, 0.90, 0.92, 0.88, 0.38, 0.40, 0.39]) # 7-class
    auc = np.array([0.89, 0.91, 0.95, 0.91, 0.51])  # 5-class
    axes2 = plt.twinx()
    axes2.plot(labels, auc, color='k', linestyle='--', marker='o')
    for i, v in enumerate(auc):
        axes2.text(i+0.2, v+0.01, str(v), ha='center', va='bottom')
    axes2.set_ylim(0, 1.0)
    axes2.set_ylabel('AUC performace')
    plt.show()
    plt.savefig('Distribution.png')


def buildDefaultData():
    print("[buildDefaultData] building...")
    t0 = time()
    X, y = loadDataFromFiles()
    dump(dict(data=X, target=y), datafile, compress='zlib')
    print(f"[buildDefaultData] done in {(time() - t0):0.3f}s")


def loadDefaultData(n_classes=7):
    print("[loadDefaultData] loading...")
    t0 = time()
    data_dict = load(datafile)
    X = data_dict['data']
    y = data_dict['target']
    if n_classes != 7:
        y = [c if c in range(1, n_classes+1) else n_classes for c in y]
    # print(pd.DataFrame(y).groupby([0]).count())
    print(f"[loadDefaultData] done in {(time() - t0):0.3f}s")
    return X, np.array(y)

def buildOriginalData(source=''):
    # print("[buildOriginalData] building...")
    t0 = time()
    X,y = loadOriDataFromFiles(source)
    dump(dict(data=X, target=y), ori_datafile, compress='zlib')
    # print(f"[buildOriginalData] done in {(time() - t0):0.3f}s")

def loadOriginalData():
    # print("[loadOriginalData] loading...")
    t0 = time()
    data_dict = load(ori_datafile)
    X = data_dict['data']
    y = data_dict['target']
    # print(f"[loadOriginalData] done in {(time() - t0):0.3f}s")
    return X, np.array(y)

def buildOriginalModel(model_name='SVM',features='C3_FE', stand=True, kbest=True):
    X, y = loadOriginalData()
    # print("[buildOriginalModel] building...")
    t0 = time()
    clf = NLoNModel(X, y, features, model_name, stand=stand, kbest=kbest, n_classes=2)
    dump(clf, ori_modelfile, compress='zlib')
    print(f"[buildOriginalModel] done in {(time() - t0):0.3f}s")


def loadOriginalModel():
    return load(ori_modelfile)


def testOriginalModel():
    model = loadOriginalModel()
    print(NLoNPredict(model, test_corpus))


def validOriginalModel(features='C3_FE'):
    # print("[validOriginalModel] start...")
    t0 = time()
    X, y = loadOriginalData()
    # print(f"[validOriginalModel] load data in {(time() - t0):0.3f}s")
    model = loadOriginalModel()
    # print(f"[validOriginalModel] load model in {(time() - t0):0.3f}s")
    ValidateModel(model, X, y,isOri=True, feature_type=features)
    print(f"[validOriginalModel] done in {(time() - t0):0.3f}s")

def plot_model_roc(n_classes=2):
    model = loadDefaultModel()
    X, y = loadDefaultData(n_classes=n_classes)
    X = NLoNFeatures.transform(X)
    if n_classes > 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0, stratify=y)
        plot_multiclass_roc(model, X_test, y_test, n_classes)
    else:
        plot_twoclass_roc(model, X, y, cv=10)

def plot_cm(n_classes=5):
    print("[plot_cm] loading...")
    t0 = time()
    model = loadDefaultModel()
    X, y = loadDefaultData(n_classes=n_classes)
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0, stratify=y)
    X_test = NLoNFeatures.transform(X_test)
    print(f"[plot_cm] transform done in {(time() - t0):0.3f}s")
    y_pred =model.predict(X_test)
    print(f"[plot_cm] predict done in {(time() - t0):0.3f}s")
    plot_confusion_matrix(y_test, y_pred, n_classes)
    print(f"[plot_cm] done in {(time() - t0):0.3f}s")

def plot_ori_model_roc():
    model = loadOriginalModel()
    X, y = loadOriginalData()
    X = NLoNFeatures.transform(X)
    plot_twoclass_roc(model, X, y, cv=10)


def buildExtendData(source=''):
    # print("[buildExtendData] building...")
    t0 = time()
    X,y = loadExtDataFromFiles(source)
    dump(dict(data=X, target=y), ext_datafile, compress='zlib')
    # print(f"[buildExtendData] done in {(time() - t0):0.3f}s")

def loadExtendData():
    # print("[loadExtendData] loading...")
    t0 = time()
    data_dict = load(ext_datafile)
    X = data_dict['data']
    y = data_dict['target']
    # print(f"[loadExtendData] done in {(time() - t0):0.3f}s")
    return X, np.array(y)

def buildExtendModel(model_name='SVM',features='C3_FE', stand=True, kbest=True):
    X, y = loadExtendData()
    # print("[buildExtendModel] building...")
    t0 = time()
    clf = NLoNModel(X, y, features, model_name, stand=stand, kbest=kbest, n_classes=12)
    dump(clf, ext_modelfile, compress='zlib')
    print(f"[buildExtendModel] done in {(time() - t0):0.3f}s")


def loadExtendModel():
    return load(ext_modelfile)


def testExtendModel():
    model = loadExtendModel()
    print(NLoNPredict(model, test_corpus))


def validExtendModel(features='C3_FE'):
    # print("[validExtendModel] start...")
    t0 = time()
    X, y = loadExtendData()
    print(f"[validExtendModel] load data in {(time() - t0):0.3f}s")
    model = loadExtendModel()
    print(f"[validExtendModel] load model in {(time() - t0):0.3f}s")
    ValidateModel(model, X, y,isOri=False, feature_type=features, cv=2)
    print(f"[validExtendModel] done in {(time() - t0):0.3f}s")