import os
from time import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split

from nlon_py.data.config import (
    DataLoaderConfig, ModelConfig,
    DEFAULT_CONFIG, ORIGINAL_CONFIG, EXTEND_CONFIG,
)
from nlon_py.data.loader import load_data_from_files
from nlon_py.data.make_data import get_category_dict
from nlon_py.features import NLoNFeatures
from nlon_py.model import (CompareModels, NLoNModel, NLoNPredict,
                           SearchParams_SVM, ValidateModel,
                           plot_multiclass_roc, plot_twoclass_roc,
                           plot_confusion_matrix)

test_corpus = [
    'This is natural language.',
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
    ''' Why does my JavaScript code receive a "No \'Access-Control-Allow-Origin\' header
     is present on the requested resource" error, while Postman does not?''',
]


# ── Unified functions ────────────────────────────────────────────────────────

def build_data(cfg: ModelConfig, source: str = '') -> None:
    loader_cfg = cfg.loader
    if source:
        loader_cfg = DataLoaderConfig(
            sources={source: loader_cfg.sources[source]},
            data_dir=loader_cfg.data_dir,
            label_col=loader_cfg.label_col,
            category_dict=loader_cfg.category_dict,
            nrows=loader_cfg.nrows,
            label_is_str=loader_cfg.label_is_str,
        )
    X, y = load_data_from_files(loader_cfg)
    dump(dict(data=X, target=y), cfg.data_file, compress='zlib')


def load_data(cfg: ModelConfig, n_classes: Optional[int] = None) -> tuple:
    data_dict = load(cfg.data_file)
    X = data_dict['data']
    y = data_dict['target']
    effective_n = n_classes if n_classes is not None else cfg.n_classes
    if effective_n < cfg.n_classes:
        y = [c if c in range(1, effective_n + 1) else effective_n for c in y]
    return X, np.array(y)


def build_model(cfg: ModelConfig, model_name: str = 'SVM', features: str = 'C3_FE',
                stand: bool = True, kbest: bool = True,
                n_classes: Optional[int] = None) -> None:
    actual_n = n_classes if n_classes is not None else cfg.n_classes
    X, y = load_data(cfg, n_classes=actual_n)
    t0 = time()
    clf = NLoNModel(X, y, features, model_name=model_name, stand=stand,
                    kbest=kbest, n_classes=actual_n)
    X_feat = NLoNFeatures.fit_transform(X, feature_type=features)
    clf.fit(X_feat, y)
    dump(clf, cfg.model_file, compress='zlib')
    print(f"[build_model:{cfg.name}] done in {(time() - t0):0.3f}s")


def load_model(cfg: ModelConfig):
    return load(cfg.model_file)


def test_model(cfg: ModelConfig) -> None:
    model = load_model(cfg)
    print(NLoNPredict(model, test_corpus))


def valid_model(cfg: ModelConfig, features: str = 'C3_FE',
                cv: Optional[int] = None, n_classes: Optional[int] = None) -> None:
    t0 = time()
    X, y = load_data(cfg, n_classes=n_classes)
    model = load_model(cfg)
    ValidateModel(model, X, y, isOri=cfg.is_binary, feature_type=features, cv=cv)
    print(f"[valid_model:{cfg.name}] done in {(time() - t0):0.3f}s")


# ── Backward-compat wrappers ─────────────────────────────────────────────────

def buildDefaultData():
    build_data(DEFAULT_CONFIG)

def loadDefaultData(n_classes=7):
    return load_data(DEFAULT_CONFIG, n_classes=n_classes)

def buildDefaultModel(n_classes=7, features='C3_FE', stand=True, kbest=True, model_name='SVM'):
    build_model(DEFAULT_CONFIG, model_name=model_name, features=features, stand=stand,
                kbest=kbest, n_classes=n_classes)

def loadDefaultModel():
    return load_model(DEFAULT_CONFIG)

def testDefaultModel():
    test_model(DEFAULT_CONFIG)

def validDefaultModel():
    valid_model(DEFAULT_CONFIG, n_classes=5)


def buildOriginalData(source=''):
    build_data(ORIGINAL_CONFIG, source=source)

def loadOriginalData():
    return load_data(ORIGINAL_CONFIG)

def buildOriginalModel(model_name='SVM', features='C3_FE', stand=True, kbest=True):
    build_model(ORIGINAL_CONFIG, model_name=model_name, features=features,
                stand=stand, kbest=kbest)

def loadOriginalModel():
    return load_model(ORIGINAL_CONFIG)

def testOriginalModel():
    test_model(ORIGINAL_CONFIG)

def validOriginalModel(features='C3_FE'):
    valid_model(ORIGINAL_CONFIG, features=features)


def buildExtendData(source=''):
    build_data(EXTEND_CONFIG, source=source)

def loadExtendData():
    return load_data(EXTEND_CONFIG)

def buildExtendModel(model_name='SVM', features='C3_FE', stand=True, kbest=True):
    build_model(EXTEND_CONFIG, model_name=model_name, features=features,
                stand=stand, kbest=kbest)

def loadExtendModel():
    return load_model(EXTEND_CONFIG)

def testExtendModel():
    test_model(EXTEND_CONFIG)

def validExtendModel(features='C3_FE'):
    valid_model(EXTEND_CONFIG, features=features, cv=2)


# ── Unchanged utility functions ──────────────────────────────────────────────

def searchParams(n_classes=7):
    print("[searchParams] start...")
    t0 = time()
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
        plt.text(i + 0.2, v + 100, str(v), ha='center', va='bottom')
    plt.ylim(0, 8000)
    plt.ylabel('Samples')
    ax.plot(labels, counts, '--')
    plt.title('Categories Distribution and AUC performance')
    auc = np.array([0.89, 0.91, 0.95, 0.91, 0.51])
    axes2 = plt.twinx()
    axes2.plot(labels, auc, color='k', linestyle='--', marker='o')
    for i, v in enumerate(auc):
        axes2.text(i + 0.2, v + 0.01, str(v), ha='center', va='bottom')
    axes2.set_ylim(0, 1.0)
    axes2.set_ylabel('AUC performace')
    plt.show()
    plt.savefig('Distribution.png')


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
    y_pred = model.predict(X_test)
    print(f"[plot_cm] predict done in {(time() - t0):0.3f}s")
    plot_confusion_matrix(y_test, y_pred, n_classes)
    print(f"[plot_cm] done in {(time() - t0):0.3f}s")


def plot_ori_model_roc():
    model = loadOriginalModel()
    X, y = loadOriginalData()
    X = NLoNFeatures.transform(X)
    plot_twoclass_roc(model, X, y, cv=10)
