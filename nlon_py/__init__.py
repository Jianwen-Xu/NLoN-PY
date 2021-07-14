"""Top-level package for nlon-py."""

__author__ = """Jianwen Xu"""
__email__ = 'xujianwen37@gmail.com'
__version__ = '0.1.2'

from nlon_py.data.build_model import buildDefaultModel, loadDefaultModel,validDefaultModel
from nlon_py.model import NLoNPredict

# buildDefaultModel()
# model = loadDefaultModel()
# print(NLoNPredict(model, ['This is natural language.',
#       'public void NotNaturalLanguageFunction(int i, String s)']))
# validDefaultModel()