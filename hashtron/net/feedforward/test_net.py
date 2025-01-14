import unittest
import os
from hashtron.net.feedforward.net import Net
from hashtron.layer.majpool2d.layer import MajPool2DLayer
from hashtron.datasets.mnist.mnist import MNISTDataset
from hashtron.datasets.squareroot.api import MediumClasses, Sample, medium
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
                pred = tron.network.infer(sample) & 1
                actual = sample.output() & 1
                if pred == actual:
                    correct+=1
            print(100 * correct // len(dataset), '% on 500', ['train', 'eval'][i], 'MNIST samples')

    def test_sqrt(self):
        # Specify network size
        fanout1 = 3
        fanout2 = 13
        fanout3 = 3
        fanout4 = 13
        # Create a Hashtron network (square root net)
        tron = Net.new()
        tron.new_layer(fanout1*fanout2*fanout3*fanout4, 0, 1<<fanout4)
        tron.new_combiner(MajPool2DLayer(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1))
        tron.new_layer(fanout1*fanout2, 0, 1<<fanout2)
        tron.new_combiner(MajPool2DLayer(fanout2, 1, fanout1, 1, fanout2, 1, 1))
        tron.new_layer(1, MediumClasses)
        # load weights from zlib file
        filename = os.path.expanduser('~/classifier/cmd/train_squareroot/output.100.json.t.zlib')
        if os.path.exists(filename):
            ok = tron.io.read_zlib_weights_from_file(filename)
        else:
            # load online model weights zlib file
            with urllib.request.urlopen('https://www.hashtron.cloud/dl/sqrt/sqrt.output.100.json.t.zlib') as f:
                with tempfile.NamedTemporaryFile(delete=True) as dst_file:
                    data = f.read()
                    dst_file.write(data)
                    ok = tron.io.read_zlib_weights_from_file(dst_file.name)
        # test the datasets
        correct = 0
        data = medium()
        for sample in data:
            pred = tron.network.infer(sample)
            actual = sample.output()
            if pred == actual:
                correct+=1
        print(100 * correct // len(data), '% on ', len(data), ' square roots')
        self.assertEqual(correct, len(data))

if __name__ == '__main__':
    unittest.main()
