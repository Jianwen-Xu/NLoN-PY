import os

from joblib import dump, load

from nlon_py.data.make_data import loadDataFromFiles
from nlon_py.features import ConvertFeatures, ComputeFeatures, TriGramsAndFeatures
from nlon_py.model import NLoNModel, NLoNPredict, ValidateModel

pwd_path = os.path.abspath(os.path.dirname(__file__))
modelfile = os.path.join(pwd_path, 'default_model.joblib')


def buildDefaultModel():
    X, y = loadDataFromFiles()
    # clf = NLoNModel(X, y, features=Character3Grams)
    # clf = NLoNModel(X, y, features=FeatureExtraction)
    clf = NLoNModel(X, y, features=TriGramsAndFeatures)
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
