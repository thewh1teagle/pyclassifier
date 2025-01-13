import unittest
from hashtron.classifier.constructor import Hashtron

class TestHashtronForward(unittest.TestCase):
    def test_forward(self):
        # Create a Hashtron instance with a sample program
        tron = Hashtron.new([[1, 4], [3, 1]], 2)

        # Call the bytes_buffer method
        result = tron.forward.forward(0, False)

        # Ensure the output is not None
        self.assertIsNotNone(result, "forward returned None")

        # Check if the expected integer is the output
        self.assertIs(0, result)


if __name__ == '__main__':
    unittest.main()
