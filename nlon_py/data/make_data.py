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

category_dict = {1: 'NL', 2: 'CODE',3:'TRACE',4:'LOG',5:'NL_CODE',6:'NL_TRACE',7:'NL_LOG'}

def get_category_dict():
    return category_dict

def loadDataFromFiles():
    X = []
    y = []
    for source, filename in filenames.items():
        data = pd.read_csv(os.path.join(pwd_path, filename),
                           header=0, encoding="UTF-8")
        data.insert(0, 'Source', source, True)
        if source == 'lucene':
            data['Text'] = list(map(lambda text: re.sub(
                r'^[>\s]+', '', text), data['Text']))
        X.extend(data['Text'])
        y.extend(data['Class'])
        data['Class'] = data['Class'].map(category_dict)
        data.to_csv(path_or_buf=os.path.join(pwd_path, f'{source}.csv'), columns=[
                    'Source', 'Text', 'Class'], index=False)
    return X, np.asarray(y)


def loadStopWords():
    stop_words_file = os.path.join(pwd_path, 'mysql_sw_wo_code_words.txt')
    stop_words = pd.read_csv(stop_words_file, header=None)
    return stop_words[0].values.tolist()
