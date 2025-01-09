import unittest
from hash.hash import Hash

class TestHash(unittest.TestCase):
    def test_hash(self):
        # Test cases for hash function
        self.assertEqual(Hash.hash(0, 0, 0), 0)
        self.assertLess(Hash.hash(1, 1, 10), 10)

    def test_string_hash(self):
        # Test cases for string_hash function
        self.assertEqual(Hash.string_hash(0, "test"), Hash.string_hash(0, "test"))

    def test_loop_length(self):
        # Loop length test
        bound1 = 10
        bound2 = 100000
        count = 0

        max_val = 1
        while max_val <= (1 << bound1):
            visited = [False] * max_val
            current = 0

            for s in range(bound2):
                current = Hash.hash(current, s, max_val)
                if current == 0 or visited[current]:
                    visited = [False] * max_val
                    continue
                else:
                    visited[current] = True
                    count += 1

            max_val <<= 1

        print(f"Tested bound1 1 << {bound1}, bound2 {bound2}, result: {count} (higher is likely better)")

if __name__ == '__main__':
    unittest.main()
