import unittest
from hashtron.layer.majpool2d.combiner import MajPool2D

class TestMajPool2D(unittest.TestCase):
    def test_put(self):
        vec = [False] * 400
        combiner = MajPool2D(vec, 10, 10, 2, 2, 5, 5, 1, 0)
        combiner.put(0, True)
        self.assertTrue(vec[0])

    def test_disregard(self):
        vec = [False] * 400
        combiner = MajPool2D(vec, 10, 10, 2, 2, 5, 5, 1, 0)
        self.assertTrue(combiner.disregard(0))

    def test_feature(self):
        vec = [True] * 400
        combiner = MajPool2D(vec, 10, 10, 2, 2, 5, 5, 1, 0)
        self.assertEqual(combiner.feature(0), 33554431)  # Example expected output

if __name__ == '__main__':
    unittest.main()
