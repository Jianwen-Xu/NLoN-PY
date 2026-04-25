# Refactor: Consolidate Data Loaders, Model Variants, Feature Classes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate 3× duplicated data loaders and model lifecycle functions by introducing `ModelConfig`/`DataLoaderConfig` dataclasses, a unified loader, and unified model functions; preserve existing public API via thin wrappers.

**Architecture:** Two new files (`config.py`, `loader.py`) hold dataclasses and the single loader function. `build_model.py` is rewritten with unified functions delegating to configs; old public names become 1-line wrappers. The unused `Features` class is removed from `features.py`. The three per-variant loader files (`make_ori_data.py`, `make_ext_data.py`, `loadDataFromFiles` in `make_data.py`) are deleted.

**Tech Stack:** Python 3.x, dataclasses, joblib, pandas, numpy, scikit-learn, pytest

**Environment constraints:**
- Work in `.worktrees/refactor-dedup/` (branch `refactor/dedup-loaders-models`)
- Run tests with: `uv run pytest tests/test_refactor.py -v` (from repo root)
- Do NOT append to `tests/test_nlon_py.py` — it cannot be collected (old deps fail at import)
- New tests go in `tests/test_refactor.py` (create if not exists)
- `uv run python -c "..."` for one-off Python checks

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `nlon_py/data/config.py` | `DataLoaderConfig`, `ModelConfig` dataclasses + 3 pre-built instances |
| Create | `nlon_py/data/loader.py` | `load_data_from_files(cfg)` — single unified CSV loader |
| Modify | `nlon_py/data/build_model.py` | Unified `build_data/load_data/build_model/load_model/test_model/valid_model` + backward-compat wrappers |
| Modify | `nlon_py/features.py` | Remove unused `Features` class |
| Modify | `nlon_py/data/make_data.py` | Remove `loadDataFromFiles` (absorbed into loader.py) |
| Delete | `nlon_py/data/original_data/make_ori_data.py` | Logic moved to loader + config |
| Delete | `nlon_py/data/extend_data/make_ext_data.py` | Logic moved to loader + config |
| Create | `tests/test_refactor.py` | Config + loader unit tests (separate file — old test file broken by import deps) |

---

### Task 1: Remove unused `Features` class from `features.py`

**Files:**
- Modify: `nlon_py/features.py`

- [ ] **Step 1: Confirm `Features` is unused**

Run:
```bash
grep -rn "Features()" nlon_py/ --include="*.py" | grep -v "FeaturesOri"
```
Expected output: no matches (only `FeaturesOri()` is instantiated, in `FeatureExtraction()`).

- [ ] **Step 2: Remove the `Features` class**

In `nlon_py/features.py`, delete lines 53–94 (the entire `Features` class, from `class Features:` through `def StartWithAt(self, text):`). The file should jump directly from `trigram_vectorizer.preprocessor = preproc` to `class FeaturesOri:`.

- [ ] **Step 3: Verify imports still work**

Run:
```bash
python -c "from nlon_py.features import FeaturesOri, NLoNFeatures, FeatureExtraction; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add nlon_py/features.py
git commit -m "refactor: remove unused Features class (FeaturesOri is the active impl)"
```

---

### Task 2: Create `nlon_py/data/config.py`

**Files:**
- Create: `nlon_py/data/config.py`
- Modify: `tests/test_nlon_py.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_refactor.py` (create the file):

```python
class TestConfig(unittest.TestCase):
    def test_default_config_n_classes(self):
        from nlon_py.data.config import DEFAULT_CONFIG
        self.assertEqual(DEFAULT_CONFIG.n_classes, 7)

    def test_original_config_is_binary(self):
        from nlon_py.data.config import ORIGINAL_CONFIG
        self.assertTrue(ORIGINAL_CONFIG.is_binary)
        self.assertEqual(ORIGINAL_CONFIG.n_classes, 2)

    def test_extend_config_n_classes(self):
        from nlon_py.data.config import EXTEND_CONFIG
        self.assertEqual(EXTEND_CONFIG.n_classes, 12)

    def test_loader_nrows(self):
        from nlon_py.data.config import DEFAULT_CONFIG, ORIGINAL_CONFIG, EXTEND_CONFIG
        self.assertIsNone(DEFAULT_CONFIG.loader.nrows)
        self.assertEqual(ORIGINAL_CONFIG.loader.nrows, 2000)
        self.assertEqual(EXTEND_CONFIG.loader.nrows, 500)

    def test_extend_label_is_str(self):
        from nlon_py.data.config import DEFAULT_CONFIG, EXTEND_CONFIG
        self.assertTrue(EXTEND_CONFIG.loader.label_is_str)
        self.assertFalse(DEFAULT_CONFIG.loader.label_is_str)

    def test_loader_label_cols(self):
        from nlon_py.data.config import DEFAULT_CONFIG, ORIGINAL_CONFIG, EXTEND_CONFIG
        self.assertEqual(DEFAULT_CONFIG.loader.label_col, 'Class')
        self.assertEqual(ORIGINAL_CONFIG.loader.label_col, 'Fabio')
        self.assertEqual(EXTEND_CONFIG.loader.label_col, 'Jianwen')
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_nlon_py.py::TestConfig -v
```
Expected: 6 errors — `ModuleNotFoundError: No module named 'nlon_py.data.config'`

- [ ] **Step 3: Create `nlon_py/data/config.py`**

```python
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
    category_dict: dict     # str→int; used only when label_is_str=True
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_nlon_py.py::TestConfig -v
```
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add nlon_py/data/config.py tests/test_nlon_py.py
git commit -m "feat: add ModelConfig and DataLoaderConfig dataclasses with pre-built configs"
```

---

### Task 3: Create `nlon_py/data/loader.py`

**Files:**
- Create: `nlon_py/data/loader.py`
- Modify: `tests/test_nlon_py.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_refactor.py` (create the file):

```python
class TestLoader(unittest.TestCase):
    def _make_df(self, label_col, labels):
        import pandas as pd
        return pd.DataFrame({'Text': ['hello', 'world'], label_col: labels})

    def test_load_int_labels(self):
        from unittest.mock import patch
        import numpy as np
        from nlon_py.data.loader import load_data_from_files
        from nlon_py.data.config import DataLoaderConfig
        cfg = DataLoaderConfig(
            sources={'s1': 'f.csv'}, data_dir='/fake',
            label_col='Class', category_dict={}, label_is_str=False,
        )
        with patch('nlon_py.data.loader.pd.read_csv',
                   return_value=self._make_df('Class', [1, 2])):
            X, y = load_data_from_files(cfg)
        self.assertEqual(X, ['hello', 'world'])
        self.assertEqual(list(y), [1, 2])

    def test_load_str_labels_mapped(self):
        from unittest.mock import patch
        from nlon_py.data.loader import load_data_from_files
        from nlon_py.data.config import DataLoaderConfig
        cfg = DataLoaderConfig(
            sources={'s1': 'f.csv'}, data_dir='/fake',
            label_col='Label', category_dict={'A': 1, 'B': 2}, label_is_str=True,
        )
        with patch('nlon_py.data.loader.pd.read_csv',
                   return_value=self._make_df('Label', ['A', 'B'])):
            X, y = load_data_from_files(cfg)
        self.assertEqual(list(y), [1, 2])

    def test_load_multiple_sources_concatenates(self):
        from unittest.mock import patch
        from nlon_py.data.loader import load_data_from_files
        from nlon_py.data.config import DataLoaderConfig
        cfg = DataLoaderConfig(
            sources={'s1': 'f1.csv', 's2': 'f2.csv'}, data_dir='/fake',
            label_col='Class', category_dict={}, label_is_str=False,
        )
        with patch('nlon_py.data.loader.pd.read_csv', side_effect=[
            self._make_df('Class', [1, 2]),
            self._make_df('Class', [3, 4]),
        ]):
            X, y = load_data_from_files(cfg)
        self.assertEqual(len(X), 4)
        self.assertEqual(list(y), [1, 2, 3, 4])

    def test_lucene_strips_quote_prefix(self):
        import pandas as pd
        from unittest.mock import patch
        from nlon_py.data.loader import load_data_from_files
        from nlon_py.data.config import DataLoaderConfig
        cfg = DataLoaderConfig(
            sources={'lucene': 'f.csv'}, data_dir='/fake',
            label_col='Class', category_dict={}, label_is_str=False,
        )
        mock_df = pd.DataFrame({'Text': ['> quoted', '> > nested'], 'Class': [1, 2]})
        with patch('nlon_py.data.loader.pd.read_csv', return_value=mock_df):
            X, y = load_data_from_files(cfg)
        self.assertEqual(X[0], 'quoted')
        self.assertEqual(X[1], 'nested')
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_nlon_py.py::TestLoader -v
```
Expected: 4 errors — `ModuleNotFoundError: No module named 'nlon_py.data.loader'`

- [ ] **Step 3: Create `nlon_py/data/loader.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_nlon_py.py::TestLoader -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add nlon_py/data/loader.py tests/test_nlon_py.py
git commit -m "feat: add unified load_data_from_files with DataLoaderConfig"
```

---

### Task 4: Rewrite `build_model.py` with unified functions + wrappers

**Files:**
- Modify: `nlon_py/data/build_model.py`
- Modify: `tests/test_nlon_py.py`

- [ ] **Step 1: Write failing tests for backward-compat wrappers**

Add to `tests/test_refactor.py` (create the file):

```python
class TestBuildModelWrappers(unittest.TestCase):
    def test_wrapper_functions_importable(self):
        from nlon_py.data.build_model import (
            buildDefaultData, loadDefaultData, buildDefaultModel, loadDefaultModel,
            testDefaultModel, validDefaultModel,
            buildOriginalData, loadOriginalData, buildOriginalModel, loadOriginalModel,
            testOriginalModel, validOriginalModel,
            buildExtendData, loadExtendData, buildExtendModel, loadExtendModel,
            testExtendModel, validExtendModel,
        )
        for fn in [buildDefaultData, loadDefaultData, buildDefaultModel, loadDefaultModel,
                   testDefaultModel, validDefaultModel, buildOriginalData, loadOriginalData,
                   buildOriginalModel, loadOriginalModel, testOriginalModel, validOriginalModel,
                   buildExtendData, loadExtendData, buildExtendModel, loadExtendModel,
                   testExtendModel, validExtendModel]:
            self.assertTrue(callable(fn))

    def test_unified_build_data_importable(self):
        from nlon_py.data.build_model import build_data, load_data, build_model, load_model
        for fn in [build_data, load_data, build_model, load_model]:
            self.assertTrue(callable(fn))
```

- [ ] **Step 2: Run tests to confirm they currently pass (wrappers already exist)**

```bash
python -m pytest tests/test_nlon_py.py::TestBuildModelWrappers -v
```
Expected: 2 PASSED (wrappers exist; `build_data` etc. will FAIL — that's fine, mark test as expected-fail or note it fails for `build_data`).

Actually the `test_unified_build_data_importable` test will fail since those names don't exist yet. That's the intended red state.

- [ ] **Step 3: Replace `nlon_py/data/build_model.py` with unified implementation**

Write the full file:

```python
import os
from time import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split

from nlon_py.data.config import (
    DataLoaderConfig, ModelConfig,
    DEFAULT_CONFIG, ORIGINAL_CONFIG, EXTEND_CONFIG,
)
from nlon_py.data.loader import load_data_from_files
from nlon_py.data.make_data import get_category_dict
from nlon_py.features import NLoNFeatures
from nlon_py.model import (CompareModels, NLoNModel, NLoNPredict,
                           SearchParams_SVM, ValidateModel,
                           plot_multiclass_roc, plot_twoclass_roc,
                           plot_confusion_matrix)

test_corpus = [
    'This is natural language.',
    'public void NotNaturalLanguageFunction(int i, String s)',
    '''Exception in thread "main" java.lang.NullPointerException
     at com.example.myproject.Book.getTitle(Book.java:16)
     at com.example.myproject.Author.getBookTitles(Author.java:25)
     at com.example.myproject.Bootstrap.main(Bootstrap.java:14)''',
    '''2012-02-02 12:47:03,309 ERROR [com.api.bg.sample] - Exception is 
     :::java.lang.IndexOutOfBoundsException: Index: 0, Size: 0''',
    '''However, I only get the file name, not the file content. When I add enctype=
     "multipart/form-data" to the <form>, then request.getParameter() returns null.''',
    '''The format is the same as getStacktrace, for e.g. 
     I/System.out(4844): java.lang.NullPointerException
     at com.temp.ttscancel.MainActivity.onCreate(MainActivity.java:43)
     at android.app.Activity.performCreate(Activity.java:5248)
     at android.app.Instrumentation.callActivityOnCreate(Instrumentation.java:1110)''',
    ''' Why does my JavaScript code receive a "No \'Access-Control-Allow-Origin\' header 
     is present on the requested resource" error, while Postman does not?''',
]


# ── Unified functions ────────────────────────────────────────────────────────

def build_data(cfg: ModelConfig, source: str = '') -> None:
    loader_cfg = cfg.loader
    if source:
        loader_cfg = DataLoaderConfig(
            sources={source: loader_cfg.sources[source]},
            data_dir=loader_cfg.data_dir,
            label_col=loader_cfg.label_col,
            category_dict=loader_cfg.category_dict,
            nrows=loader_cfg.nrows,
            label_is_str=loader_cfg.label_is_str,
        )
    X, y = load_data_from_files(loader_cfg)
    dump(dict(data=X, target=y), cfg.data_file, compress='zlib')


def load_data(cfg: ModelConfig, n_classes: Optional[int] = None) -> tuple:
    data_dict = load(cfg.data_file)
    X = data_dict['data']
    y = data_dict['target']
    effective_n = n_classes if n_classes is not None else cfg.n_classes
    if effective_n < cfg.n_classes:
        y = [c if c in range(1, effective_n + 1) else effective_n for c in y]
    return X, np.array(y)


def build_model(cfg: ModelConfig, model_name: str = 'SVM', features: str = 'C3_FE',
                stand: bool = True, kbest: bool = True,
                n_classes: Optional[int] = None) -> None:
    actual_n = n_classes if n_classes is not None else cfg.n_classes
    X, y = load_data(cfg, n_classes=actual_n)
    t0 = time()
    clf = NLoNModel(X, y, features, model_name=model_name, stand=stand,
                    kbest=kbest, n_classes=actual_n)
    dump(clf, cfg.model_file, compress='zlib')
    print(f"[build_model:{cfg.name}] done in {(time() - t0):0.3f}s")


def load_model(cfg: ModelConfig):
    return load(cfg.model_file)


def test_model(cfg: ModelConfig) -> None:
    model = load_model(cfg)
    print(NLoNPredict(model, test_corpus))


def valid_model(cfg: ModelConfig, features: str = 'C3_FE',
                cv: Optional[int] = None, n_classes: Optional[int] = None) -> None:
    t0 = time()
    X, y = load_data(cfg, n_classes=n_classes)
    model = load_model(cfg)
    ValidateModel(model, X, y, isOri=cfg.is_binary, feature_type=features, cv=cv)
    print(f"[valid_model:{cfg.name}] done in {(time() - t0):0.3f}s")


# ── Backward-compat wrappers ─────────────────────────────────────────────────

def buildDefaultData():
    build_data(DEFAULT_CONFIG)

def loadDefaultData(n_classes=7):
    return load_data(DEFAULT_CONFIG, n_classes=n_classes)

def buildDefaultModel(n_classes=7, features='C3_FE', stand=True, kbest=True):
    build_model(DEFAULT_CONFIG, features=features, stand=stand, kbest=kbest,
                n_classes=n_classes)

def loadDefaultModel():
    return load_model(DEFAULT_CONFIG)

def testDefaultModel():
    test_model(DEFAULT_CONFIG)

def validDefaultModel():
    valid_model(DEFAULT_CONFIG, n_classes=5)


def buildOriginalData(source=''):
    build_data(ORIGINAL_CONFIG, source=source)

def loadOriginalData():
    return load_data(ORIGINAL_CONFIG)

def buildOriginalModel(model_name='SVM', features='C3_FE', stand=True, kbest=True):
    build_model(ORIGINAL_CONFIG, model_name=model_name, features=features,
                stand=stand, kbest=kbest)

def loadOriginalModel():
    return load_model(ORIGINAL_CONFIG)

def testOriginalModel():
    test_model(ORIGINAL_CONFIG)

def validOriginalModel(features='C3_FE'):
    valid_model(ORIGINAL_CONFIG, features=features)


def buildExtendData(source=''):
    build_data(EXTEND_CONFIG, source=source)

def loadExtendData():
    return load_data(EXTEND_CONFIG)

def buildExtendModel(model_name='SVM', features='C3_FE', stand=True, kbest=True):
    build_model(EXTEND_CONFIG, model_name=model_name, features=features,
                stand=stand, kbest=kbest)

def loadExtendModel():
    return load_model(EXTEND_CONFIG)

def testExtendModel():
    test_model(EXTEND_CONFIG)

def validExtendModel(features='C3_FE'):
    valid_model(EXTEND_CONFIG, features=features, cv=2)


# ── Unchanged utility functions ──────────────────────────────────────────────

def searchParams(n_classes=7):
    print("[searchParams] start...")
    t0 = time()
    X, y = loadDefaultData(n_classes=n_classes)
    SearchParams_SVM(X, y)
    print(f"[searchParams] done in {(time() - t0):0.3f}s")


def compareDifModels(n_classes, cv=None):
    print("[compareDifModels] start...")
    t0 = time()
    X, y = loadDefaultData(n_classes)
    CompareModels(X, y, cv=cv)
    print(f"[compareDifModels] done in {(time() - t0):0.3f}s")


def plotDistribution(n_classes=7):
    X, y = loadDefaultData(n_classes)
    class_dict = get_category_dict()
    unique, counts = np.unique(y, return_counts=True)
    labels = [class_dict[x] for x in unique]
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.bar(labels, counts, width=0.5)
    for i, v in enumerate(counts):
        plt.text(i + 0.2, v + 100, str(v), ha='center', va='bottom')
    plt.ylim(0, 8000)
    plt.ylabel('Samples')
    ax.plot(labels, counts, '--')
    plt.title('Categories Distribution and AUC performance')
    auc = np.array([0.89, 0.91, 0.95, 0.91, 0.51])
    axes2 = plt.twinx()
    axes2.plot(labels, auc, color='k', linestyle='--', marker='o')
    for i, v in enumerate(auc):
        axes2.text(i + 0.2, v + 0.01, str(v), ha='center', va='bottom')
    axes2.set_ylim(0, 1.0)
    axes2.set_ylabel('AUC performace')
    plt.show()
    plt.savefig('Distribution.png')


def plot_model_roc(n_classes=2):
    model = loadDefaultModel()
    X, y = loadDefaultData(n_classes=n_classes)
    X = NLoNFeatures.transform(X)
    if n_classes > 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0, stratify=y)
        plot_multiclass_roc(model, X_test, y_test, n_classes)
    else:
        plot_twoclass_roc(model, X, y, cv=10)


def plot_cm(n_classes=5):
    print("[plot_cm] loading...")
    t0 = time()
    model = loadDefaultModel()
    X, y = loadDefaultData(n_classes=n_classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y)
    X_test = NLoNFeatures.transform(X_test)
    print(f"[plot_cm] transform done in {(time() - t0):0.3f}s")
    y_pred = model.predict(X_test)
    print(f"[plot_cm] predict done in {(time() - t0):0.3f}s")
    plot_confusion_matrix(y_test, y_pred, n_classes)
    print(f"[plot_cm] done in {(time() - t0):0.3f}s")


def plot_ori_model_roc():
    model = loadOriginalModel()
    X, y = loadOriginalData()
    X = NLoNFeatures.transform(X)
    plot_twoclass_roc(model, X, y, cv=10)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_nlon_py.py::TestBuildModelWrappers -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add nlon_py/data/build_model.py tests/test_nlon_py.py
git commit -m "refactor: unify build_model.py with ModelConfig-based functions; wrappers preserve public API"
```

---

### Task 5: Remove old loader files and `loadDataFromFiles` from `make_data.py`

**Files:**
- Modify: `nlon_py/data/make_data.py`
- Delete: `nlon_py/data/original_data/make_ori_data.py`
- Delete: `nlon_py/data/extend_data/make_ext_data.py`

- [ ] **Step 1: Confirm no remaining imports of the old loaders**

Run:
```bash
grep -rn "loadDataFromFiles\|loadOriDataFromFiles\|loadExtDataFromFiles\|make_ori_data\|make_ext_data" nlon_py/ --include="*.py"
```
Expected: zero matches (build_model.py no longer imports them after Task 4).

- [ ] **Step 2: Remove `loadDataFromFiles` from `make_data.py`**

In `nlon_py/data/make_data.py`, delete the `loadDataFromFiles` function (lines 25–40 in the current file). Leave `loadStopWords`, `get_category_dict`, `plotDistribution` unchanged.

The file after edit should start with:
```python
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pwd_path = os.path.abspath(os.path.dirname(__file__))

filenames = {
    'mozilla': 'lines.10k.cfo.sample.2000 - Mozilla (Firefox, Core, OS).csv',
    'kubernetes': 'lines.10k.cfo.sample.2000 - Kubernetes (Slackarchive.io).csv',
    'lucene': 'lines.10k.cfo.sample.2000 - Lucene-dev mailing list.csv',
    'bitcoin': 'lines.10k.cfo.sample.2000 - Bitcoin (github.com).csv'
}

category_dict = {1: 'NL', 2: 'CODE', 3: 'TRACE',
                 4: 'LOG', 5: 'NL_CODE', 6: 'NL_TRACE', 7: 'NL_LOG'}


def get_category_dict():
    return category_dict


def loadStopWords():
    stop_words_file = os.path.join(pwd_path, 'mysql_sw_wo_code_words.txt')
    stop_words = pd.read_csv(stop_words_file, header=None)
    return stop_words[0].values.tolist()


def plotDistribution():
    # (leave this function body exactly as it appears in the current file — do not modify it)
```

**Important:** Delete ONLY the `loadDataFromFiles` function block. Leave `get_category_dict`, `loadStopWords`, `plotDistribution`, and all `import` statements and module-level variables (`pwd_path`, `filenames`, `category_dict`) untouched.

- [ ] **Step 3: Delete the old loader files**

```bash
rm nlon_py/data/original_data/make_ori_data.py
rm nlon_py/data/extend_data/make_ext_data.py
```

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove loadDataFromFiles and per-variant loader files; logic in loader.py"
```

---

### Task 6: Integration verification

**Files:** read-only

- [ ] **Step 1: Import smoke test**

```bash
python -c "
from nlon_py.data.build_model import (
    buildDefaultData, buildDefaultModel, loadDefaultModel,
    buildOriginalData, buildOriginalModel, loadOriginalModel,
    buildExtendData, buildExtendModel, loadExtendModel,
)
from nlon_py.data.config import DEFAULT_CONFIG, ORIGINAL_CONFIG, EXTEND_CONFIG
from nlon_py.data.loader import load_data_from_files
from nlon_py.features import NLoNFeatures, FeaturesOri
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 2: Verify pre-trained model still predicts (requires existing .joblib)**

```bash
python -c "
from nlon_py.data.build_model import loadDefaultModel
from nlon_py.model import NLoNPredict
model = loadDefaultModel()
y = NLoNPredict(model, ['This is natural language.', 'public void foo(int i)'])
print(y)
assert y[0] == 'NL', f'Expected NL, got {y[0]}'
print('Prediction OK')
"
```
Expected: prints labels and `Prediction OK`

- [ ] **Step 3: Run full test suite one final time**

```bash
python -m pytest tests/ -v
```
Expected: all tests pass, no import errors.

- [ ] **Step 4: Commit spec + plan docs**

```bash
git add docs/
git commit -m "docs: add refactor spec and implementation plan"
```
