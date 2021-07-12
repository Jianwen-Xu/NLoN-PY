"""Top-level package for nlon-py."""

__author__ = """Jianwen Xu"""
__email__ = 'xujianwen37@gmail.com'
__version__ = '0.1.1'

from nlon_py.data.build_model import buildDefaultModel, loadDefaultModel
from nlon_py.data.make_data import loadDataFromFiles
from nlon_py.features import TriGramsAndFeatures, ComputeFeatures, ConvertFeatures, Character3Grams, Character3GramsForTest
from nlon_py.model import CompareModels, NLoNModel, NLoNPredict, SearchParameters_KNN

# X, y = loadDataFromFiles()
# SearchParameters_KNN(ConvertFeatures(
#     ComputeFeatures(X, TriGramsAndFeatures)), y)
# CompareModels(Character3Grams(X), y)
# ConvertFeatures(ComputeFeatures(X, Character3Grams))
# buildDefaultModel()
model = loadDefaultModel()
print(NLoNPredict(model, ['This is natural language.',
      'public void NotNaturalLanguageFunction(int i, String s)']))
