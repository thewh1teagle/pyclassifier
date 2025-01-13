import unittest
import os
from net.feedforward.net import Net
from layer.majpool2d.layer import MajPool2DLayer
from datasets.mnist.mnist import MNISTDataset
import urllib.request
import tempfile

class TestNetConstruct(unittest.TestCase):
    def test_construct(self):
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
            for sample in dataset[0:500]:
                pred = tron.network.infer(sample).feature(0) & 1
                actual = sample.output().feature(0) & 1
                if pred == actual:
                    correct+=1
            print(100 * correct // len(dataset), '% on 500', ['train', 'eval'][i], 'MNIST samples')

if __name__ == '__main__':
    unittest.main()
