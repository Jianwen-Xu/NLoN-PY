# Refactor: Consolidate Duplicate Data Loaders, Model Variants, and Feature Classes

**Date:** 2026-04-24  
**Status:** Approved

## Context

NLoN-PY has three near-identical data loading functions (`loadDataFromFiles`, `loadOriDataFromFiles`, `loadExtDataFromFiles`), three sets of near-identical model lifecycle functions in `build_model.py` (build/load/validate for default, original, extend variants), and two feature classes (`Features`, `FeaturesOri`) with identical method signatures but slightly different implementations.

Goal: eliminate duplication, make adding new variants a config-only change, preserve the existing public API in `__init__.py`.

---

## Architecture

### New Files

**`nlon_py/data/config.py`**  
Defines `DataLoaderConfig` and `ModelConfig` dataclasses, plus three pre-built config instances.

**`nlon_py/data/loader.py`**  
Single `load_data_from_files(cfg: DataLoaderConfig)` replaces the three per-variant loader functions.

### Modified Files

**`nlon_py/data/build_model.py`**  
Unified functions (`build_data`, `load_data`, `build_model`, `load_model`, `test_model`, `valid_model`) replace 3× duplicated blocks. Old public names become 1-liner wrappers delegating to config-based functions.

**`nlon_py/features.py`**  
Verify `Features` vs `FeaturesOri` usage. Merge into single class (using `FeaturesOri` impl, which is what `FeatureExtraction` and `NLoNFeatures` call). Remove or alias `Features` if unused.

### Effectively Deleted

- `nlon_py/data/original_data/make_ori_data.py` — logic absorbed into `loader.py` + config
- `nlon_py/data/extend_data/make_ext_data.py` — same
- `loadDataFromFiles` in `nlon_py/data/make_data.py` — same (rest of file kept: `loadStopWords`, `get_category_dict`, `plotDistribution`)

---

## Data Structures

```python
# nlon_py/data/config.py
from dataclasses import dataclass
from typing import Optional
import os

DATA_DIR = os.path.abspath(os.path.dirname(__file__))
ORI_DIR  = os.path.join(DATA_DIR, 'original_data')
EXT_DIR  = os.path.join(DATA_DIR, 'extend_data')

DEFAULT_FILENAMES = {
    'mozilla':    'lines.10k.cfo.sample.2000 - Mozilla (Firefox, Core, OS).csv',
    'kubernetes': 'lines.10k.cfo.sample.2000 - Kubernetes (Slackarchive.io).csv',
    'lucene':     'lines.10k.cfo.sample.2000 - Lucene-dev mailing list.csv',
    'bitcoin':    'lines.10k.cfo.sample.2000 - Bitcoin (github.com).csv',
}
ORI_FILENAMES = {k: v for k, v in DEFAULT_FILENAMES.items() if k != 'bitcoin'}
EXT_FILENAMES = ORI_FILENAMES  # same 3 sources

@dataclass
class DataLoaderConfig:
    sources: dict           # {source_name: filename}
    data_dir: str           # base directory for CSV files
    label_col: str          # CSV column holding labels
    category_dict: dict     # str→int map; only used when label_is_str=True
    nrows: Optional[int] = None
    label_is_str: bool = False  # True → map str labels via category_dict (extend only)

@dataclass
class ModelConfig:
    name: str
    n_classes: int
    data_file: str          # absolute path to .joblib for data cache
    model_file: str         # absolute path to .joblib for model
    loader: DataLoaderConfig
    feature_type: str = 'C3_FE'
    is_binary: bool = False # maps to isOri param in ValidateModel

# Pre-built config instances
DEFAULT_CATEGORY_DICT  = {1: 'NL', 2: 'CODE', 3: 'TRACE', 4: 'LOG',
                           5: 'NL_CODE', 6: 'NL_TRACE', 7: 'NL_LOG'}
ORIGINAL_CATEGORY_DICT = {1: 'NL', 2: 'Not'}
EXTEND_CATEGORY_DICT   = {'NL': 1, 'Code': 2, 'Error': 3, 'Trace': 4, 'Log': 5,
                           'Math': 6, 'URL': 7, 'File': 8, 'Id': 9,
                           'Version': 10, 'Other': 11, 'Mixed': 12}

DEFAULT_CONFIG = ModelConfig(
    name='default', n_classes=7,
    data_file=os.path.join(DATA_DIR, 'default_data.joblib'),
    model_file=os.path.join(DATA_DIR, 'default_model.joblib'),
    loader=DataLoaderConfig(
        sources=DEFAULT_FILENAMES, data_dir=DATA_DIR,
        label_col='Class', category_dict=DEFAULT_CATEGORY_DICT,
    ),
)

ORIGINAL_CONFIG = ModelConfig(
    name='original', n_classes=2, is_binary=True,
    data_file=os.path.join(DATA_DIR, 'original_data.joblib'),
    model_file=os.path.join(DATA_DIR, 'original_model.joblib'),
    loader=DataLoaderConfig(
        sources=ORI_FILENAMES, data_dir=ORI_DIR,
        label_col='Fabio', category_dict=ORIGINAL_CATEGORY_DICT, nrows=2000,
    ),
)

EXTEND_CONFIG = ModelConfig(
    name='extend', n_classes=12,
    data_file=os.path.join(DATA_DIR, 'extend_data.joblib'),
    model_file=os.path.join(DATA_DIR, 'extend_model.joblib'),
    loader=DataLoaderConfig(
        sources=EXT_FILENAMES, data_dir=EXT_DIR,
        label_col='Jianwen', category_dict=EXTEND_CATEGORY_DICT,
        nrows=500, label_is_str=True,
    ),
)
```

---

## Unified Loader (`nlon_py/data/loader.py`)

```python
def load_data_from_files(cfg: DataLoaderConfig) -> tuple[list, np.ndarray]:
    X, y = [], []
    for source, filename in cfg.sources.items():
        data = pd.read_csv(os.path.join(cfg.data_dir, filename),
                           header=0, encoding='UTF-8', nrows=cfg.nrows)
        data.insert(0, 'Source', source, True)
        if source == 'lucene':
            data['Text'] = data['Text'].map(lambda t: re.sub(r'^[>\s]+', '', t))
        X.extend(data['Text'])
        raw_labels = data[cfg.label_col]
        if cfg.label_is_str:
            y.extend(raw_labels.map(cfg.category_dict))
        else:
            y.extend(raw_labels)
    return X, np.asarray(y)
```

---

## Unified Model Functions (`nlon_py/data/build_model.py`)

```python
def build_data(cfg: ModelConfig, source: str = '') -> None
    # if source != '', filter cfg.loader.sources to that key only; then load and dump to cfg.data_file

def load_data(cfg: ModelConfig, n_classes: Optional[int] = None) -> tuple
    # load from cfg.data_file; if n_classes < cfg.n_classes, collapse tail classes

def build_model(cfg: ModelConfig, model_name='SVM', stand=True, kbest=True) -> None
    # load_data → NLoNModel → dump to cfg.model_file

def load_model(cfg: ModelConfig) -> Pipeline
    # joblib.load(cfg.model_file)

def test_model(cfg: ModelConfig) -> None
    # load_model(cfg) → NLoNPredict(model, test_corpus)

def valid_model(cfg: ModelConfig, cv=None) -> None
    # load_data + load_model → ValidateModel(isOri=cfg.is_binary)
```

### Backward-Compat Wrappers (unchanged signatures)

```python
def buildDefaultData():               build_data(DEFAULT_CONFIG)
def loadDefaultData(n_classes=7):     return load_data(DEFAULT_CONFIG, n_classes)
def buildDefaultModel(**kw):          build_model(DEFAULT_CONFIG, **kw)
def loadDefaultModel():               return load_model(DEFAULT_CONFIG)
def testDefaultModel():               test_model(DEFAULT_CONFIG)
def validDefaultModel():              valid_model(DEFAULT_CONFIG)

def buildOriginalData(source=''):     build_data(ORIGINAL_CONFIG, source=source)
def loadOriginalData():               return load_data(ORIGINAL_CONFIG)
def buildOriginalModel(**kw):         build_model(ORIGINAL_CONFIG, **kw)
def loadOriginalModel():              return load_model(ORIGINAL_CONFIG)
def testOriginalModel():              test_model(ORIGINAL_CONFIG)
def validOriginalModel(**kw):         valid_model(ORIGINAL_CONFIG, **kw)

def buildExtendData(source=''):       build_data(EXTEND_CONFIG, source=source)
def loadExtendData():                 return load_data(EXTEND_CONFIG)
def buildExtendModel(**kw):           build_model(EXTEND_CONFIG, **kw)
def loadExtendModel():                return load_model(EXTEND_CONFIG)
def testExtendModel():                test_model(EXTEND_CONFIG)
def validExtendModel(**kw):           valid_model(EXTEND_CONFIG, **kw)
```

Source filter: `buildOriginalData(source='')` and `buildExtendData(source='')` currently accept a source filter. `build_data(cfg, source='')` handles this — when `source != ''`, loader uses only that key from `cfg.loader.sources`.

---

## Feature Class Consolidation (`nlon_py/features.py`)

**Pre-condition:** Grep `model.py` and all other files for `Features()` instantiation to confirm `Features` (non-Ori) is unused. If unused, remove it. If used, keep as alias: `Features = FeaturesOri`.

Implementation: `FeatureExtraction()` already uses `FeaturesOri`. No functional change — just delete `Features` class (or alias) after confirming zero external usage.

---

## Verification

```bash
# 1. Import smoke test
python -c "from nlon_py import (loadDefaultModel, loadOriginalModel, loadExtendModel,
    buildDefaultData, buildOriginalData, buildExtendData); print('imports OK')"

# 2. Prediction test (requires pre-built .joblib models)
python -c "from nlon_py import testDefaultModel; testDefaultModel()"

# 3. Existing test suite
python -m pytest tests/ -v
```

New variant can be added by: defining one `DataLoaderConfig` + one `ModelConfig`, adding 6 wrapper functions. Zero changes to loader or model logic.
