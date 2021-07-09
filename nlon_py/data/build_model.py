from nlon_py.model import NLoNModel
from nlon_py.data.make_data import loadDataFromFiles

X, y = loadDataFromFiles()
clf = NLoNModel(X, y, nfolds=10)
print(clf)
