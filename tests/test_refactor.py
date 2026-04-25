import os
import unittest


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


class TestLoader(unittest.TestCase):
    def _make_df(self, label_col, labels):
        import pandas as pd
        return pd.DataFrame({'Text': ['hello', 'world'], label_col: labels})

    def test_load_int_labels(self):
        from unittest.mock import patch
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


if __name__ == '__main__':
    unittest.main()
