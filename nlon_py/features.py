from sklearn.feature_extraction.text import CountVectorizer

from nlon_py.data.make_data import loadStopWords

stop_words_list = loadStopWords()


def Character3Grams(text):
    trigram_vectorizer = CountVectorizer(
        ngram_range=(3, 3), stop_words=preprocess(stop_words_list))
    return trigram_vectorizer.fit_transform(text).toarray()


def preprocess(text):
    vectorizer = CountVectorizer()
    analyze = vectorizer.build_analyzer()
    result = []
    for x in text:
        result.extend(analyze(x))
    return result
