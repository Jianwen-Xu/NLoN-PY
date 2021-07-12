import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer

from nlon_py.data.make_data import loadStopWords

pwd_path = os.path.abspath(os.path.dirname(__file__))
vecfile = os.path.join(pwd_path, 'data/default_vectorizer.joblib')

stop_words_list = loadStopWords()


def preprocess(text):
    vectorizer = CountVectorizer()
    analyze = vectorizer.build_analyzer()
    result = []
    for x in text:
        result.extend(analyze(x))
    return result


trigram_vectorizer = CountVectorizer(ngram_range=(
    3, 3), stop_words=preprocess(stop_words_list))


def Character3Grams(text):
    data = pd.DataFrame.sparse.from_spmatrix(trigram_vectorizer.fit_transform(text))
    dump(trigram_vectorizer, vecfile)
    return data


def Character3GramsForTest(text):
    vectorizer = load(vecfile)
    return vectorizer.transform(text)


def FeatureExtraction(text):
    return pd.DataFrame(text)


def ConvertFeatures(data):
    if isinstance(data, list):
        data = pd.DataFrame(data)
    if isinstance(data, pd.DataFrame):
        data = np.asarray(data)
    return data


def ComputeFeatures(text, features):
    if callable(features):
        return features(text)
    elif isinstance(features, list):
        return [feature(text) for feature in features if callable(features)]


def TriGramsAndFeatures(text):
    return pd.concat([Character3Grams(text), FeatureExtraction(text)])
