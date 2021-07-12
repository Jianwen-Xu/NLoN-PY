<<<<<<< HEAD
import os

from joblib import dump, load

from nlon_py.data.make_data import loadDataFromFiles
from nlon_py.features import Character3Grams
from nlon_py.model import NLoNModel, NLoNPredict

pwd_path = os.path.abspath(os.path.dirname(__file__))
modelfile = os.path.join(pwd_path, 'default_model.joblib')


def buildDefaultModel():
    X, y = loadDataFromFiles()
    clf = NLoNModel(X, y, features=Character3Grams)
    dump(clf, modelfile)


def loadDefaultModel():
    return load(modelfile)
=======
from nlon_py.model import NLoNModel
from nlon_py.data.make_data import loadDataFromFiles

X, y = loadDataFromFiles()
clf = NLoNModel(X, y, nfolds=10)
print(clf)
>>>>>>> e89b9f91a8c615791562fd339e67b2e83b5e7837
