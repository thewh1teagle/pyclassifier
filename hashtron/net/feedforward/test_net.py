import unittest
import os
from hashtron.net.feedforward.net import Net
from hashtron.layer.majpool2d.layer import MajPool2DLayer
from hashtron.layer.full.layer import FullLayer
from hashtron.datasets.mnist.mnist import load_mnist
from hashtron.datasets.squareroot.api import MediumClasses, Sample, medium
import urllib.request
import tempfile

class TestNetConstruct(unittest.TestCase):
    def test_construct(self):
        # Specify network size
        fanout1 = 1
        fanout2 = 5
        fanout3 = 1
        fanout4 = 4
        fanout5 = 1
        fanout6 = 4
        # Create a Hashtron network (MNIST handwritten digits net)
        tron = Net.new()
        tron.new_layer(fanout1*fanout2*fanout3*fanout4*fanout5*fanout6, 0, 1<<(fanout6*fanout6*2//3))
        tron.new_combiner(MajPool2DLayer(fanout1*fanout2*fanout3*fanout4*fanout6, 1, fanout5, 1, fanout6, 1, 1))
        tron.new_layer(fanout1*fanout2*fanout3*fanout4, 0, 1<<(fanout4*fanout4*2//3))
        tron.new_combiner(MajPool2DLayer(fanout1*fanout2*fanout4, 1, fanout3, 1, fanout4, 1, 1))
        tron.new_layer(fanout1*fanout2, 0, 1<<(fanout2*fanout2*2//3))
        tron.new_combiner(FullLayer(fanout2, 1, 1))
        # load weights from zlib file
        filename = os.path.expanduser('~/classifier/cmd/train_mnist/output.78.json.zlib')
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
        
        # test the datasets
        for i in range(2):
            # load offline then online mnist
            try:
                dataset, _, _, _, = load_mnist('~/pyclassifier/datasets/mnist/')
            except FileNotFoundError:
                dataset, _, _, _, = load_mnist()
            
            correct = 0
            for sample in dataset[0:500]:
                pred = tron.network.infer(sample) % 10
                actual = sample.output() % 10
                if pred == actual:
                    correct+=1
            print(100 * correct // len(dataset), '% on 500', ['train', 'eval'][i], 'MNIST samples')

    def test_sqrt(self):
        # Specify network size
        fanout1 = 3
        fanout2 = 12
        fanout3 = 3
        fanout4 = 12
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
            with urllib.request.urlopen('https://hashtron.cloud/dl/classifier_models_v0.1/infer_squareroot.100.json.t.zlib') as f:
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
