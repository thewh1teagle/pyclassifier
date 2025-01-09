# Hashtron Network Implementation

This project implements a hashtron network with feedforward layers and combiners,
including a 2D majority pooling layer. The network supports feedforward inference
with hashtron-based classifiers but not training. For training see the original
golang version which is CPU optimized.

## Modules

- `hash`: Implements fast modular hash function.
- `hashtron`: Implements hashtron classifier.
- `layer`: Defines layer and combiner interfaces, including a 2D majority pooling layer.
- `net`: Implements the feedforward network and related utilities.

## Usage

To use the network, create an instance of `FeedforwardNetwork` called e.g. `net` and add layers
using `net.new_layer`, adding `net.new_combiner` in between layers.
Load your model from ZLIB file using `net.io.read_zlib_weights_from_file(file_name)`.
Use `net.network.infer(input_number_or_sample)` to perform inference.

## Examples

```python
from pyclassifier.net.feedforward.net import Net
from pyclassifier.layer.majpool2d.layer import MajPool2DLayer
from pyclassifier.datasets.mnist.mnist import MNISTDataset
import urllib.request
import tempfile
import os

# Specify network size
fanout1 = 3
fanout2 = 5
fanout3 = 3
fanout4 = 5
# Create a Hashtron network (MNIST handwritten digits net)
tron = Net.new()
tron.new_layer(fanout1*fanout2*fanout3*fanout4, 0, 1<<fanout4)
tron.new_combiner(MajPool2DLayer(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1))
tron.new_layer(fanout1*fanout2, 0, 1<<fanout2)
tron.new_combiner(MajPool2DLayer(fanout2, 1, fanout1, 1, fanout2, 1, 1))
tron.new_layer(1, 0)
# load weights from zlib file
filename = os.path.expanduser('~/classifier/cmd/train_mnist/output.77.json.t.zlib')
if os.path.exists(filename):
    ok = tron.io.read_zlib_weights_from_file(filename)
else:
    # load online model weights zlib file
    with urllib.request.urlopen('https://www.hashtron.cloud/dl/mnist/output.77.json.t.zlib') as f:
        with tempfile.NamedTemporaryFile(delete=True) as dst_file:
            data = f.read()
            dst_file.write(data)
            ok = tron.io.read_zlib_weights_from_file(dst_file.name)

if not ok:
    raise Exception('model weights were not loaded')

# test the datasets
for i in range(2):
    # load offline then online mnist
    try:
        dataset = MNISTDataset(True, i == 0)
    except FileNotFoundError:
        dataset = MNISTDataset(True, i == 0, MNISTDataset.download())
    
    correct = 0
    for sample in dataset:
        pred = tron.network.infer(sample).feature(0) & 1
        actual = sample.output().feature(0) & 1
        if pred == actual:
            correct+=1
    print(100 * correct // len(dataset), '% on', len(dataset), 'MNIST samples')
```

### Summary

This Python translation maintains the structure and functionality of the original
Go code, adapting Go-specific features to Python equivalents. The translation
includes classes for hash functions, hashtron classifiers, layers, and a
feedforward network, along with unit tests and a README for documentation.

### Contributing

0. Open issue
1. Fork the repo
2. Implement what is needed (no blobs in repo, host datasets or demo models externally)
3. Run testcases: `find . -name 'test_*py' -exec python -m unittest '{}' \;`
4. Contribute a pull request

### License

MIT
