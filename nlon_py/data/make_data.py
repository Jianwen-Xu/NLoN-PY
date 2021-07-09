import re

import pandas as pd
import numpy as np

filenames = {
    'mozilla': 'data/lines.10k.cfo.sample.2000 - Mozilla (Firefox, Core, OS).csv',
    'kubernetes': 'data/lines.10k.cfo.sample.2000 - Kubernetes (Slackarchive.io).csv',
    'lucene': 'data/lines.10k.cfo.sample.2000 - Lucene-dev mailing list.csv',
    'bitcoin': 'data/lines.10k.cfo.sample.2000 - Bitcoin (github.com).csv'
}


def loadDataFromFiles():
    X = []
    y = []
    for source, filename in filenames.items():
        data = pd.read_csv(filename, header=0, encoding="ISO-8859-1")
        if source == 'lucene':
            data['Text'] = list(map(lambda text: re.sub(
                r'^[>\s]+', '', text), data['Text']))
        X.extend(data['Text'])
        y.extend(data['Class'])

    return X, np.array(y)
