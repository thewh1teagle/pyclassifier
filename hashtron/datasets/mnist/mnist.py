import gzip
import os
import hashlib
import random
from typing import List, Tuple
from hashtron.hash.hash import Hash
import tempfile
import urllib.request
from hashtron.datasets.stringhash.bytehash import ByteSample
from io import BytesIO

# Constants
ImgSize = 28
SmallImgSize = 13

class Input:
    """Represents an original 28x28 MNIST image."""
    def __init__(self, image: List[int], label: int):
        self.buf = bytes(image)
        self.label = label

    def feature(self, n: int) -> int:
        return ByteSample(self.buf, self.label).feature(n)

    def parity(self) -> int:
        return 0

    def output(self) -> int:
        return self.label

class SmallInput:
    """Represents a downsampled 13x13 MNIST image."""
    def __init__(self, image: List[int], label: int):
        self.buf = bytes(image)
        self.label = label

    def feature(self, n: int) -> int:
        return ByteSample(self.buf, self.label).feature(n)

    def parity(self) -> int:
        return 0

    def output(self) -> int:
        return self.label

def download_mnist() -> str:
    """Downloads MNIST files to a temporary directory."""
    dst_dir = tempfile.gettempdir()
    base_url = 'https://www.hashtron.cloud/dl/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    for file in files:
        url = base_url + file
        dest = os.path.join(dst_dir, file)
        if not os.path.exists(dest):
            with urllib.request.urlopen(url) as response:
                data = response.read()
                with open(dest, 'wb') as f:
                    f.write(data)
    return dst_dir

def _load_file(file_name: str, expected_hash: str, dataset_dir: str = None) -> bytes:
    """Loads and verifies a gzipped MNIST file."""
    search_dirs = [
        '/tmp/mnist/',
        os.path.expanduser('~/pyclassifier/datasets/mnist/')
    ]
    if dataset_dir:
        search_dirs.insert(0, dataset_dir)
    for dir_path in search_dirs:
        file_path = os.path.join(dir_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = f.read()
                file_hash = hashlib.sha256(data).hexdigest()
                if file_hash != expected_hash:
                    raise ValueError(f"Hash mismatch for {file_path}")
                with gzip.GzipFile(fileobj=BytesIO(data)) as gz:
                    return gz.read()
    raise FileNotFoundError(f"{file_name} not found in {search_dirs}")

def load_images(file_name: str, expected_hash: str, dataset_dir: str) -> Tuple[List[List[int]], List[List[int]]]:
    """Loads images and generates original and downsampled versions."""
    data = _load_file(file_name, expected_hash, dataset_dir)
    image_data = data[16:]  # Skip header
    num_images = len(image_data) // (ImgSize * ImgSize)
    original = []
    small = []
    for i in range(num_images):
        # Extract original image
        start = i * ImgSize * ImgSize
        original_img = list(image_data[start: start + ImgSize * ImgSize])
        original.append(original_img)
        # Generate small image
        cropped = []
        # Crop to 26x26 (remove 1px border)
        for y in range(1, ImgSize - 1):
            row = image_data[start + y*ImgSize + 1 : start + (y+1)*ImgSize - 1]
            cropped.extend(row)
        # Downsample to 13x13 using max pooling
        small_img = []
        for y in range(SmallImgSize):
            for x in range(SmallImgSize):
                offset = 2 * (y * (ImgSize - 2) + x)
                block = [
                    cropped[offset],
                    cropped[offset + 1],
                    cropped[offset + (ImgSize - 2)],
                    cropped[offset + (ImgSize - 2) + 1]
                ]
                small_img.append(max(block))
        small.append(small_img)
    return original, small

def load_labels(file_name: str, expected_hash: str, dataset_dir: str) -> List[int]:
    """Loads MNIST labels."""
    data = _load_file(file_name, expected_hash, dataset_dir)
    return list(data[8:])  # Skip header

def load_mnist(dataset_dir: str = None) -> Tuple[List[Input], List[Input], List[SmallInput], List[SmallInput]]:
    """Loads all MNIST datasets (original and small, train and test)."""
    if not dataset_dir:
        # Download files if not found
        dataset_dir = download_mnist()
    # File hashes
    hashes = {
        'train-images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
        'train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
        't10k-images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
        't10k-labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'
    }
    # Load training data
    train_orig, train_small = load_images('train-images-idx3-ubyte.gz', hashes['train-images-idx3-ubyte.gz'], dataset_dir)
    train_labels = load_labels('train-labels-idx1-ubyte.gz', hashes['train-labels-idx1-ubyte.gz'], dataset_dir)
    # Load test data
    test_orig, test_small = load_images('t10k-images-idx3-ubyte.gz', hashes['t10k-images-idx3-ubyte.gz'], dataset_dir)
    test_labels = load_labels('t10k-labels-idx1-ubyte.gz', hashes['t10k-labels-idx1-ubyte.gz'], dataset_dir)
    # Create samples
    train_set = [Input(img, lbl) for img, lbl in zip(train_orig, train_labels)]
    small_train_set = [SmallInput(img, lbl) for img, lbl in zip(train_small, train_labels)]
    test_set = [Input(img, lbl) for img, lbl in zip(test_orig, test_labels)]
    small_test_set = [SmallInput(img, lbl) for img, lbl in zip(test_small, test_labels)]
    return train_set, test_set, small_train_set, small_test_set

# Example usage:
# train, test, small_train, small_test = load_mnist()
