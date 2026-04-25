=====
Usage
=====

Predict with the default model
-------------------------------

Load a pre-built model and classify text::

    from nlon_py.data.build_model import loadDefaultModel
    from nlon_py.model import NLoNPredict

    model = loadDefaultModel()

    text = [
        'This is natural language.',
        'public void NotNaturalLanguageFunction(int i, String s)',
    ]

    print(NLoNPredict(model, text))
    # e.g. ['NL', 'CODE']

``NLoNPredict`` accepts raw strings — feature extraction is handled internally.

Available model variants
------------------------

Three pre-built model variants are provided:

+----------+-------------------------------------------+--------------------------------------------+
| Variant  | Load function                             | Classes                                    |
+==========+===========================================+============================================+
| default  | ``loadDefaultModel()``                    | NL, CODE, TRACE, LOG, NL_CODE, NL_TRACE,   |
|          |                                           | NL_LOG                                     |
+----------+-------------------------------------------+--------------------------------------------+
| original | ``loadOriginalModel()``                   | NL, Not                                    |
+----------+-------------------------------------------+--------------------------------------------+
| extend   | ``loadExtendModel()``                     | NL, Code, Error, Trace, Log, Math, URL,    |
|          |                                           | File, Id, Version, Other, Mixed            |
+----------+-------------------------------------------+--------------------------------------------+

Build / retrain models
----------------------

Build data cache then train::

    from nlon_py.data.build_model import buildDefaultData, buildDefaultModel

    buildDefaultData()                    # load CSVs → default_data.joblib
    buildDefaultModel(model_name='SVM')   # train    → default_model.joblib

Original and extend variants follow the same pattern::

    from nlon_py.data.build_model import (
        buildOriginalData, buildOriginalModel,
        buildExtendData,   buildExtendModel,
    )

    buildOriginalData(source='mozilla')   # single source or '' for all three
    buildOriginalModel(model_name='glmnet', features='FE', stand=False, kbest=False)

    buildExtendData()
    buildExtendModel(model_name='glmnet', features='C3_FE', stand=False, kbest=False)

Validate a trained model
------------------------

::

    from nlon_py.data.build_model import validDefaultModel, validOriginalModel

    validDefaultModel()
    validOriginalModel()

Advanced: config and unified loader
-------------------------------------

The refactored internals expose ``DataLoaderConfig`` / ``ModelConfig`` dataclasses
and a single ``load_data_from_files`` loader.  Custom variants can be added by
defining one config instance and six wrapper functions — no changes to loader or
model logic required::

    from nlon_py.data.config import DataLoaderConfig, ModelConfig
    from nlon_py.data.loader import load_data_from_files

    cfg = DataLoaderConfig(
        sources={'my_source': 'my_data.csv'},
        data_dir='/path/to/data',
        label_col='Label',
        category_dict={1: 'NL', 2: 'CODE'},
    )
    X, y = load_data_from_files(cfg)
