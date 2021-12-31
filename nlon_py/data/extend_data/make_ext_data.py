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

category_dict = {'NL': 1, 'Code': 2, 'Error': 3, 'Trace': 4, 'Log': 5, 'Math': 6,
                 'URL': 7, 'File': 8,  'Id': 9,  'Version': 10,  'Other': 11,  'Mixed': 12}


def get_category_dict():
    return category_dict


def loadExtDataFromFiles(source):
    X = []
    y = []
    files = {}
    if source != '':
        files = {source:filenames[source]}
    else:
        files = filenames
    for source, filename in files.items():
        data = pd.read_csv(os.path.join(pwd_path, filename),
                           header=0, encoding="UTF-8", nrows=500)
        data.insert(0, 'Source', source, True)
        if source == 'lucene':
            data['Text'] = list(map(lambda text: re.sub(
                r'^[>\s]+', '', text), data['Text']))
        X.extend(data['Text'])
        y.extend(data['Jianwen'].map(category_dict))
        data['Class'] = data['Jianwen']
        data.to_csv(path_or_buf=os.path.join(pwd_path, f'{source}.csv'), columns=[
                    'Source', 'Text', 'Class'], index=False)
    return X, np.asarray(y)

