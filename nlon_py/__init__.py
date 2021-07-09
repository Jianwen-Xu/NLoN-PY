"""Top-level package for nlon-py."""

__author__ = """Jianwen Xu"""
__email__ = 'xujianwen37@gmail.com'
__version__ = '0.1.1'

from nlon_py.model import NLoNModel
from nlon_py.data.make_data import loadDataFromFiles
from nlon_py.features import Character3Grams
X, y = loadDataFromFiles()
clf = NLoNModel(Character3Grams(X), y, nfolds=10)
print(clf)
