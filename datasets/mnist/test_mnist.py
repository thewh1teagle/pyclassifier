from datasets.mnist.mnist import MNISTDataset
import unittest

class TestMnist(unittest.TestCase):
    def test_mnist(self):
        try:
            dataset = MNISTDataset()
            dataset1 = MNISTDataset(False, False)
            dataset2 = MNISTDataset(True, False)
            dataset3 = MNISTDataset(False, True)
        except FileNotFoundError:
            try:
                dd = MNISTDataset.download()
                dataset = MNISTDataset(dataset_dir=dd)
                dataset1 = MNISTDataset(False, False, dataset_dir=dd)
                dataset2 = MNISTDataset(True, False, dataset_dir=dd)
                dataset3 = MNISTDataset(False, True, dataset_dir=dd)
            except FileNotFoundError:
                print("MNIST not available, skipping test...")
                return

        self.assertTrue(len(dataset) == 60000)
        self.assertTrue(len(dataset1) == 10000)
        self.assertTrue(len(dataset2) == 10000)
        self.assertTrue(len(dataset3) == 60000)

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

        # Shuffle the dataset
        dataset.shuffle()

if __name__ == '__main__':
    unittest.main()
