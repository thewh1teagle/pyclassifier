from hashtron.datasets.mnist.mnist import load_mnist
import unittest

class TestMnist(unittest.TestCase):
    def test_mnist(self):
        try:
            dataset, dataset1, dataset2, dataset3 = load_mnist('~/pyclassifier/datasets/mnist/')
        except FileNotFoundError:
            try:
                dataset, dataset1, dataset2, dataset3 = load_mnist()
            except FileNotFoundError:
                print("MNIST not available, skipping test...")
                return

        self.assertTrue(len(dataset) == 60000)
        self.assertTrue(len(dataset1) == 10000)
        self.assertTrue(len(dataset2) == 60000)
        self.assertTrue(len(dataset3) == 10000)

        # Access the first sample
        sample = dataset[0]
        print(f"Label: {sample.label}, Feature 123: {sample.feature(123)}")
        # Access the first sample
        sample = dataset1[0]
        print(f"Label: {sample.label}, Feature 123: {sample.feature(123)}")
        # Access the first sample
        sample = dataset2[0]
        print(f"Label: {sample.label}, Feature 123: {sample.feature(123)}")
        # Access the first sample
        sample = dataset3[0]
        print(f"Label: {sample.label}, Feature 123: {sample.feature(123)}")

        # Iterate over the dataset
        for sample in dataset:
            print(f"Label: {sample.label}, Feature 0: {sample.feature(0)}")
            break


if __name__ == '__main__':
    unittest.main()
