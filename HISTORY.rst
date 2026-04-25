=======
History
=======

0.2 (2026-04-25)
------------------

* Refactor: consolidate three near-identical data loaders into a single
  ``load_data_from_files(cfg)`` parameterised by ``DataLoaderConfig``
* Refactor: replace 3× duplicated model lifecycle blocks with six unified
  functions (``build_data``, ``load_data``, ``build_model``, ``load_model``,
  ``test_model``, ``valid_model``) plus backward-compatible camelCase wrappers
* Add ``nlon_py/data/config.py`` (``DataLoaderConfig``, ``ModelConfig``,
  ``DEFAULT_CONFIG``, ``ORIGINAL_CONFIG``, ``EXTEND_CONFIG``)
* Add ``nlon_py/data/loader.py`` (unified CSV loader)
* Remove unused ``Features`` class (``FeaturesOri`` is the active implementation)
* Remove per-variant loader files ``make_ori_data.py`` and ``make_ext_data.py``
* Convert ``__init__.py`` to lazy imports to avoid loading heavy ML deps at
  import time
* All existing public API names preserved (no breaking changes)

0.1.4 (2021-07-18)
------------------

* Implement basic model and functions

0.1.0 (2021-06-10)
------------------

* First release on PyPI.
