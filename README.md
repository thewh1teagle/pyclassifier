# Hashtron Network Implementation

This project implements a hashtron network with feedforward layers and combiners,
including a 2D majority pooling layer. The network supports feedforward inference
with hashtron-based classifiers but not training. For training, see the original
golang version which is CPU and GPU (CUDA) optimized.

## Modules

- `datasets`: Loader for demo datasets such as MNIST.
- `hash`: Implements fast modular hash function.
- `classifier`: Implements the hashtron classifier which repeatedly calls the hash.
- `layer`: Defines layer and combiner interfaces, including a 2D majority pooling combiner layer.
- `net`: Implements the feedforward network and related utilities.

## Usage

To use the network, create an instance of `FeedforwardNetwork` called e.g. `net` and add layers
using `net.new_layer`, adding `net.new_combiner` in between layers.
Load your model from ZLIB file using `net.io.read_zlib_weights_from_file(file_name)`.
Use `net.network.infer(input_number_or_sample)` to perform inference.

## Examples

`pip install hashtron`

```python
from hashtron.net.feedforward.net import Net
from hashtron.layer.majpool2d.layer import MajPool2DLayer
from hashtron.layer.full.layer import FullLayer
from hashtron.datasets.mnist.mnist import load_mnist
import urllib.request
import tempfile
import os
import random

# Specify network size
fanout1 = 1
fanout2 = 5
fanout3 = 1
fanout4 = 4
fanout5 = 1
fanout6 = 4
# Create a Hashtron network (MNIST handwritten digits net)
tron = Net.new()
tron.new_layer(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6, 0, 1<<((fanout6*fanout6*2)//3))
tron.new_combiner(MajPool2DLayer(fanout1*fanout2*fanout3*fanout4*fanout6, 1, fanout5, 1, fanout6, 1, 1))
tron.new_layer(fanout1*fanout2*fanout3*fanout4, 0, 1<<((fanout4*fanout4*2)//3))
tron.new_combiner(MajPool2DLayer(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1))
tron.new_layer(fanout1*fanout2, 0, 1<<((fanout2*fanout2*2)//3))
tron.new_combiner(FullLayer(fanout2, 1, 1))
# load weights from zlib file
filename = os.path.expanduser('~/classifier/cmd/train_mnist/output.78.json.t.zlib')
if os.path.exists(filename):
    ok = tron.io.read_zlib_weights_from_file(filename)
else:
    # load online model weights zlib file
    with urllib.request.urlopen('https://hashtron.cloud/dl/classifier_models_v0.1/infer_mnist.78.json.t.zlib') as f:
        with tempfile.NamedTemporaryFile(delete=True) as dst_file:
            data = f.read()
            dst_file.write(data)
            ok = tron.io.read_zlib_weights_from_file(dst_file.name)

if not ok:
    raise Exception('model weights were not loaded')

# load offline then online mnist
try:
    dataset, _, _, _, = load_mnist('~/pyclassifier/datasets/mnist/')
except FileNotFoundError:
    dataset, _, _, _, = load_mnist()

random.shuffle(dataset)

correct = 0
for sample in dataset:
    pred = tron.network.infer(sample) % 10
    actual = sample.output() % 10
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

1. Open issue
2. Fork the repo
3. Implement what is needed (no blobs in repo, host datasets or demo models externally)
4. Add tests as needed
5. Experimentally install the package: `pip install -e .`
6. Run all testcases: `pip install pytest` and then `python3 -m pytest`
7. Contribute a pull request

### License

MIT
