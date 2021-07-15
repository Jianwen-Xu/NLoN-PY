import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load

from nlon_py.data.make_data import loadDataFromFiles, get_category_dict
from nlon_py.features import (ComputeFeatures, ConvertFeatures,
                              TriGramsAndFeatures)
from nlon_py.model import NLoNModel, NLoNPredict, ValidateModel

pwd_path = os.path.abspath(os.path.dirname(__file__))
modelfile = os.path.join(pwd_path, 'default_model.joblib')


def buildDefaultModel():
    X, y = loadDataFromFiles()
    # clf = NLoNModel(X, y, features=Character3Grams)
    # clf = NLoNModel(X, y, features=FeatureExtraction)
    clf = NLoNModel(X, y, features=TriGramsAndFeatures,
                    model_name='Linear SVM')
    dump(clf, modelfile)


def loadDefaultModel():
    return load(modelfile)


def testDefaultModel():
    model = loadDefaultModel()
    print(NLoNPredict(model, ['This is natural language.',
                              'public void NotNaturalLanguageFunction(int i, String s)']))


def validDefaultModel():
    X, y = loadDataFromFiles()
    X = ConvertFeatures(ComputeFeatures(X, TriGramsAndFeatures))
    model = loadDefaultModel()
    ValidateModel(model, X, y)


def plotDistribution():
    X, y = loadDataFromFiles()
    class_dict = get_category_dict()
    unique, counts = np.unique(y, return_counts=True)
    labels = [class_dict[x] for x in unique]
    plt.bar(labels, counts, width=0.5)
    for i, v in enumerate(counts):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.title('Categories Distribution')
    plt.savefig('Distribution.png')
