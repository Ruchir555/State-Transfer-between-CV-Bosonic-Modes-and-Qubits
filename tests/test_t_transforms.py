import unittest
from math import atanh

from t_transforms import is_majorized, t_parameters, truncated_tmss_probabilities


class TTransformTests(unittest.TestCase):
    def test_majorization_definition(self):
        self.assertTrue(is_majorized([5, 3, 2], [6, 3, 1]))
        self.assertFalse(is_majorized([6, 3, 1], [5, 3, 2]))
        self.assertFalse(is_majorized([0.5, 0.5], [0.6, 0.3]))

    def test_worked_example_matches_notebook(self):
        x = [5, 3, 2]
        y = [6, 3, 1]
        x_before = x.copy()
        y_before = y.copy()

        parameters, indices = t_parameters(x, y)

        self.assertEqual(x, x_before)
        self.assertEqual(y, y_before)
        self.assertEqual(indices, [2, 2])
        self.assertAlmostEqual(parameters[0], 2.0 / 3.0)
        self.assertAlmostEqual(parameters[1], 2.0 / 3.0)

    def test_invalid_transform_is_rejected(self):
        with self.assertRaises(ValueError):
            t_parameters([6, 3, 1], [5, 3, 2])
        with self.assertRaises(ValueError):
            t_parameters([0.5, 0.5], [1.0])

    def test_truncated_tmss_can_be_normalized(self):
        values = truncated_tmss_probabilities(atanh(0.75), 5, renormalize=True)
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(sum(values), 1.0)
        self.assertTrue(all(left > right for left, right in zip(values, values[1:])))


if __name__ == "__main__":
    unittest.main()
