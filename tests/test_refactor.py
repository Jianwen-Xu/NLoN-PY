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


if __name__ == '__main__':
    unittest.main()
