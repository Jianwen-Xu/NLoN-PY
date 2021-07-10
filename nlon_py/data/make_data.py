import os
import re

import numpy as np
import pandas as pd

pwd_path = os.path.abspath(os.path.dirname(__file__))

filenames = {
    'mozilla': 'lines.10k.cfo.sample.2000 - Mozilla (Firefox, Core, OS).csv',
    'kubernetes': 'lines.10k.cfo.sample.2000 - Kubernetes (Slackarchive.io).csv',
    'lucene': 'lines.10k.cfo.sample.2000 - Lucene-dev mailing list.csv',
    'bitcoin': 'lines.10k.cfo.sample.2000 - Bitcoin (github.com).csv'
}


def loadDataFromFiles():
    X = []
    y = []
    for source, filename in filenames.items():
        data = pd.read_csv(os.path.join(pwd_path, filename),
                           header=0, encoding="ISO-8859-1")
        if source == 'lucene':
            data['Text'] = list(map(lambda text: re.sub(
                r'^[>\s]+', '', text), data['Text']))
        X.extend(data['Text'])
        y.extend(data['Class'])

    return X, np.array(y)


def loadStopWords():
    stop_words_file = os.path.join(pwd_path, 'mysql_sw_wo_code_words.txt')
    stop_words = pd.read_csv(stop_words_file, header=None)
    return stop_words[0].values.tolist()
