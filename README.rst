=======
nlon-py
=======


.. image:: https://img.shields.io/pypi/v/nlon-py.svg
        :target: https://pypi.python.org/pypi/nlon-py
.. image:: https://app.travis-ci.com/Jianwen-Xu/NLoN-PY.svg?branch=master
        :target: https://app.travis-ci.com/Jianwen-Xu/NLoN-PY
.. image:: https://readthedocs.org/projects/nlon-py/badge/?version=latest
        :target: https://nlon-py.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Python package for identifying text containing natural language (or not) using machine learning.


* Free software: MIT license
* Documentation: https://nlon-py.readthedocs.io.


Features
--------

* Three model variants: **default** (7-class), **original** (binary NL/Not), **extend** (12-class)
* Single unified data loader and model lifecycle — new variants require only a config definition
* Pre-built ``.joblib`` models included; retrain from source CSVs with one function call
* ``NLoNFeatures`` supports three feature sets: ``FE`` (feature engineering), ``C3`` (character 3-gram), ``C3_FE`` (combined)
* Compatible with SVM, glmnet, Naive Bayes, Nearest Neighbors, and XGBoost classifiers

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
