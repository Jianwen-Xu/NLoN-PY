import os
from dataclasses import dataclass
from typing import Optional

DATA_DIR = os.path.abspath(os.path.dirname(__file__))
ORI_DIR = os.path.join(DATA_DIR, 'original_data')
EXT_DIR = os.path.join(DATA_DIR, 'extend_data')

_DEFAULT_FILENAMES = {
    'mozilla':    'lines.10k.cfo.sample.2000 - Mozilla (Firefox, Core, OS).csv',
    'kubernetes': 'lines.10k.cfo.sample.2000 - Kubernetes (Slackarchive.io).csv',
    'lucene':     'lines.10k.cfo.sample.2000 - Lucene-dev mailing list.csv',
    'bitcoin':    'lines.10k.cfo.sample.2000 - Bitcoin (github.com).csv',
}
_ORI_EXT_FILENAMES = {k: v for k, v in _DEFAULT_FILENAMES.items() if k != 'bitcoin'}


@dataclass
class DataLoaderConfig:
    sources: dict
    data_dir: str
    label_col: str
    category_dict: dict     # str->int; used only when label_is_str=True
    nrows: Optional[int] = None
    label_is_str: bool = False


@dataclass
class ModelConfig:
    name: str
    n_classes: int
    data_file: str
    model_file: str
    loader: DataLoaderConfig
    feature_type: str = 'C3_FE'
    is_binary: bool = False


DEFAULT_CONFIG = ModelConfig(
    name='default',
    n_classes=7,
    data_file=os.path.join(DATA_DIR, 'default_data.joblib'),
    model_file=os.path.join(DATA_DIR, 'default_model.joblib'),
    loader=DataLoaderConfig(
        sources=_DEFAULT_FILENAMES,
        data_dir=DATA_DIR,
        label_col='Class',
        category_dict={},
    ),
)

ORIGINAL_CONFIG = ModelConfig(
    name='original',
    n_classes=2,
    is_binary=True,
    data_file=os.path.join(DATA_DIR, 'original_data.joblib'),
    model_file=os.path.join(DATA_DIR, 'original_model.joblib'),
    loader=DataLoaderConfig(
        sources=_ORI_EXT_FILENAMES,
        data_dir=ORI_DIR,
        label_col='Fabio',
        category_dict={},
        nrows=2000,
    ),
)

EXTEND_CONFIG = ModelConfig(
    name='extend',
    n_classes=12,
    data_file=os.path.join(DATA_DIR, 'extend_data.joblib'),
    model_file=os.path.join(DATA_DIR, 'extend_model.joblib'),
    loader=DataLoaderConfig(
        sources=_ORI_EXT_FILENAMES,
        data_dir=EXT_DIR,
        label_col='Jianwen',
        category_dict={
            'NL': 1, 'Code': 2, 'Error': 3, 'Trace': 4, 'Log': 5,
            'Math': 6, 'URL': 7, 'File': 8, 'Id': 9,
            'Version': 10, 'Other': 11, 'Mixed': 12,
        },
        nrows=500,
        label_is_str=True,
    ),
)
