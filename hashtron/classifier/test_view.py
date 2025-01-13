import unittest
from hashtron.classifier.constructor import Hashtron

class TestHashtronSerialize(unittest.TestCase):
    def test_serialize(self):
        # Create a Hashtron instance with a sample program
        tron = Hashtron.new([[1, 2], [3, 4]], 2)
        
        a = tron.view.write_json()

        tron.view.read_json(a)
        
        b = tron.view.write_json()
        
        self.assertEqual(a, b)

if __name__ == '__main__':
    unittest.main()
