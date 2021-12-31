"""Top-level package for nlon-py."""

__author__ = """Jianwen Xu"""
__email__ = 'xujianwen37@gmail.com'
__version__ = '0.1.5'

from types import new_class
from nlon_py.data.build_model import (buildDefaultData,buildOriginalData, buildDefaultModel,
                                      compareDifModels, loadDefaultData,loadOriginalData,
                                      plot_cm, plot_model_roc,plot_ori_model_roc,
                                      plotDistribution, searchParams,validOriginalModel,
                                      testDefaultModel, validDefaultModel,buildOriginalModel,
                                      buildExtendData, validExtendModel,buildExtendModel)
from nlon_py.features import NLoNFeatures,FeaturesOri
# from nlon_py.data.make_data import plotDistribution
# buildDefaultModel(n_classes=5)
# buildDefaultModel(n_classes=5, features='C3', stand=False)
# buildDefaultModel(n_classes=5, features='FE', kbest=False)
# buildDefaultModel(n_classes=5,features='C3_FE', kbest=False)
# buildDefaultModel(n_classes=5,features='C3_FE', kbest=True)
# buildDefaultModel(n_classes=5,features='C3_FE', stand=False)
# testDefaultModel()
# validDefaultModel()
# compareDifModels(n_classes=5)
# buildDefaultData()
# loadDefaultData(n_class=2)
# for model in ['glmnet', 'Naive Bayes', 'Nearest Neighbors', 'SVM', 'XGB']:
# # for model in ['glmnet']:    
#     print('- build model: '+ model)
#     for source in ['mozilla','kubernetes','lucene']:
#         print('-- build source: '+ source)
#         buildOriginalData(source)
#         for feature in ['FE', 'C3', 'C3_FE']:
#         # for feature in ['FE']:
#             print('--- build features: '+ feature)
#             buildOriginalModel(model_name=model, features=feature, stand=False, kbest=False)
#             validOriginalModel(features=feature)
# plot_ori_model_roc()
# plotDistribution(n_classes=5)
# searchParams(n_classes=5)
# plot_model_roc(n_classes=5)
# plot_cm()
# X, y = loadOriginalData()
# X = X[3:4]
# print(X)
# fo = FeaturesOri()
# print('token1-----------------------')
# X_t1 = fo.StopwordsRatio1(text=X[0])
# print(X_t1)

# print('token2-----------------------')
# X_t2 = fo.StopwordsRatio2(text=X[0])
# print(X_t2)

# print('FE-----------------------')
# X_FE = NLoNFeatures.fit_transform(X,feature_type='FE')
# print(X_FE)
for model in ['glmnet']:  
    for source in ['mozilla','kubernetes','lucene']:
        print('-- build source: '+ source)
        buildExtendData(source)
        for feature in ['FE', 'C3', 'C3_FE']:
            print('--- build features: '+ feature)
            buildExtendModel(model_name=model, features=feature, stand=False, kbest=False)
            validExtendModel(features=feature)