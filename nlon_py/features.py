from sklearn.feature_extraction.text import CountVectorizer


def Character3Grams(text):

    trigram_vectorizer = CountVectorizer(
        ngram_range=(3, 3), token_pattern=r'\b\w+\b', min_df=1)
    analyze = trigram_vectorizer.build_analyzer()
    return trigram_vectorizer.fit_transform(text).toarray()
