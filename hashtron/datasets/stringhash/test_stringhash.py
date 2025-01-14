from hashtron.datasets.stringhash.stringhash import Sample, BalancedSample
import unittest

class TestStringHash(unittest.TestCase):
    def test_string_hash(self):
        self.assertEqual(Sample("hello").feature(0), 1496225408)
        self.assertEqual(Sample("hello").parity(), 0)

        self.assertEqual(BalancedSample("world").feature(0), 478492469)
        self.assertEqual(BalancedSample("world").parity(), 1887138210)

if __name__ == '__main__':
    unittest.main()
