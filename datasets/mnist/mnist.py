import gzip
import os
import hashlib
import random
from typing import List
from pyclassifier.hash.hash import Hash
import tempfile
import urllib.request

class MNISTLabel:
    def __init__(self, label: int):
        self.label = label
    
    def feature(self, n: int) -> int:
        """
        Extract the n-th feature from the image.

        Args:
            n (int): The feature index.

        Returns:
            int: The extracted feature value (uint32).
        """
        return self.label

    @staticmethod
    def parity() -> int:
        """
        Used to negate the prediction output for this sample, in order to balance unbalanced datasets.
        Usual values are 0 and 1.
        Returns:
            int: The parity value (uint32).
        """
        return 0

class MNISTSample:
    def __init__(self, image: List[int], label: int):
        """
        Represents a single MNIST sample.

        Args:
            image (List[int]): The 28x28 image as a flattened list of pixels.
            label (int): The label for the image.
        """
        self.image = image
        self.label = label

    def feature(self, n: int) -> int:
        """
        Extract the n-th feature from the image.

        Args:
            n (int): The feature index.

        Returns:
            int: The extracted feature value (uint32).
        """
        size = len(self.image)
        feature_value = 0
        for j in range(4):
            idx = Hash.hash(n, j, size - 1)
            feature_value ^= int(self.image[idx]) << (8 * j)
        return int(feature_value) & 0xFFFFFFFF

    @staticmethod
    def parity() -> int:
        """
        Used to negate the prediction output for this sample, in order to balance unbalanced datasets.
        Usual values are 0 and 1.
        Returns:
            int: The parity value (uint32).
        """
        return 0

    def output(self) -> MNISTLabel:
        return MNISTLabel(self.label)
    

class MNISTDataset:
    def __init__(self, is_small: bool = True, is_train: bool = True, dataset_dir: str = None):
        """
        Initialize the MNIST dataset loader (training data only).

        Args:
            dataset_dir (str): Directory containing the MNIST dataset files.
                              If None, searches in default directories.
        """
        self.ImgSize = 28
        self.SmallImgSize = 13
        # Default search directories
        self.search_directories = [
            '/tmp/mnist/',
            os.path.expanduser('~/pyclassifier/datasets/mnist/')
        ]
        if dataset_dir:
            self.search_directories.insert(0, dataset_dir)
        if is_train:
            # File names and their expected SHA-256 hashes
            self.prefix = 'train-'
            self.files = {
                'images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
                'labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'
            }
        else:
            # File names and their expected SHA-256 hashes
            self.prefix = 't10k-'
            self.files = {
                'images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
                'labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6',
            }
        # Data storage
        self.samples: List[MNISTSample] = []
        # Load training dataset
        self._load_datasets(is_small)

    @staticmethod
    def download() -> str:
        dst_dir = tempfile.gettempdir()
        for dataset in [
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz',
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
        ]:
            with urllib.request.urlopen('https://www.hashtron.cloud/dl/mnist/' + dataset) as f:
                data = f.read()
                with open(dst_dir + '/' + dataset, 'w+b') as dst_file:
                    dst_file.write(data)
        return dst_dir

    def _load_file(self, file_name: str, expected_hash: str) -> bytes:
        """
        Load a file and verify its SHA-256 hash.

        Args:
            file_name (str): Name of the file to load.
            expected_hash (str): Expected SHA-256 hash of the file.

        Returns:
            bytes: Uncompressed file content.
        """
        for dir_path in self.search_directories:
            file_path = os.path.join(dir_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                    f.seek(0)
                    # Verify SHA-256 hash
                    file_hash = hashlib.sha256(data).hexdigest()
                    if file_hash != expected_hash:
                        raise ValueError(f"File hash mismatch for {file_path}")
                    # Unzip the file
                    with gzip.GzipFile(fileobj=f) as gz:
                        return gz.read()
        raise FileNotFoundError(f"File not found: {file_name}")


    def _load_images(self, file_name: str, expected_hash: str, is_small: bool) -> List[List[int]]:
        """
        Load and process image data.

        Args:
            file_name (str): Name of the image file.
            expected_hash (str): Expected SHA-256 hash of the file.
            is_small (bool): Whether to downsample the images.

        Returns:
            List[List[int]]: List of images as flattened lists of pixels.
        """
        data = self._load_file(file_name, expected_hash)
        # Skip header bytes (16 bytes for images)
        image_data = data[16:]

        # Calculate the number of images
        num_images = len(image_data) // (self.ImgSize * self.ImgSize)
        images = []

        for i in range(num_images):
            # Extract the image
            start = i * self.ImgSize * self.ImgSize
            end = start + self.ImgSize * self.ImgSize
            img = list(image_data[start:end])

            if is_small:
                # Crop the image by removing 1 pixel from each side
                cropped_img = []
                for y in range(1, self.ImgSize - 1):
                    row_start = y * self.ImgSize + 1
                    row_end = row_start + self.ImgSize - 2
                    cropped_img.extend(img[row_start:row_end])

                # Downsample the cropped image to 13x13 using max pooling
                small_img = []
                for y in range(self.SmallImgSize):
                    for x in range(self.SmallImgSize):
                        # Extract 2x2 block
                        block = [
                            cropped_img[(y * 2) * (self.ImgSize - 2) + (x * 2)],
                            cropped_img[(y * 2) * (self.ImgSize - 2) + (x * 2) + 1],
                            cropped_img[(y * 2 + 1) * (self.ImgSize - 2) + (x * 2)],
                            cropped_img[(y * 2 + 1) * (self.ImgSize - 2) + (x * 2) + 1],
                        ]
                        # Take the maximum value in the block
                        small_img.append(max(block))
                images.append(small_img)
            else:
                # Return the original image as a flattened list
                images.append(img)

        return images

    def _load_labels(self, file_name: str, expected_hash: str) -> List[int]:
        """
        Load label data.

        Args:
            file_name (str): Name of the label file.
            expected_hash (str): Expected SHA-256 hash of the file.

        Returns:
            List[int]: List of labels.
        """
        data = self._load_file(file_name, expected_hash)
        # Skip header bytes (8 bytes for labels)
        label_data = data[8:]
        return list(label_data)

    def _load_datasets(self, small: bool):
        """Load the training dataset."""
        images = self._load_images(self.prefix + 'images-idx3-ubyte.gz', self.files['images-idx3-ubyte.gz'], small)
        labels = self._load_labels(self.prefix + 'labels-idx1-ubyte.gz', self.files['labels-idx1-ubyte.gz'])
        # Create MNISTSample objects
        self.samples = [MNISTSample(img, label) for img, label in zip(images, labels)]

    def shuffle(self):
        """Shuffle the dataset."""
        random.shuffle(self.samples)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> MNISTSample:
        """Retrieve the nth sample from the dataset."""
        return self.samples[idx]

    def __iter__(self):
        """Iterate over the dataset, yielding MNISTSample objects."""
        return iter(self.samples)

