import os
import re

import numpy as np
import pandas as pd

from nlon_py.data.config import DataLoaderConfig


def load_data_from_files(cfg: DataLoaderConfig):
    X = []
    y = []
    for source, filename in cfg.sources.items():
        data = pd.read_csv(
            os.path.join(cfg.data_dir, filename),
            header=0, encoding='UTF-8', nrows=cfg.nrows,
        )
        data.insert(0, 'Source', source, True)
        if source == 'lucene':
            data['Text'] = data['Text'].map(lambda t: re.sub(r'^[>\s]+', '', t))
        X.extend(data['Text'])
        if cfg.label_is_str:
            y.extend(data[cfg.label_col].map(cfg.category_dict))
        else:
            y.extend(data[cfg.label_col])
    return X, np.asarray(y)
