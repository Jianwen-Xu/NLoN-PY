import os
import re

import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd

pwd_path = os.path.abspath(os.path.dirname(__file__))

filenames = {
    'mozilla': 'lines.10k.cfo.sample.2000 - Mozilla (Firefox, Core, OS).csv',
    'kubernetes': 'lines.10k.cfo.sample.2000 - Kubernetes (Slackarchive.io).csv',
    'lucene': 'lines.10k.cfo.sample.2000 - Lucene-dev mailing list.csv'
}

category_dict = {1: 'NL', 2: 'Not'}


def get_category_dict():
    return category_dict


def loadOriDataFromFiles(source):
    X = []
    y = []
    files = {}
    if source != '':
        files = {source:filenames[source]}
    else:
        files = filenames
    for source, filename in files.items():
        data = pd.read_csv(os.path.join(pwd_path, filename),
                           header=0, encoding="UTF-8", nrows=2000)
        data.insert(0, 'Source', source, True)
        if source == 'lucene':
            data['Text'] = list(map(lambda text: re.sub(
                r'^[>\s]+', '', text), data['Text']))
        X.extend(data['Text'])
        y.extend(data['Fabio'])
        data['Class'] = data['Fabio'].map(category_dict)
        data.to_csv(path_or_buf=os.path.join(pwd_path, f'{source}.csv'), columns=[
                    'Source', 'Text', 'Class'], index=False)
    return X, np.asarray(y)

