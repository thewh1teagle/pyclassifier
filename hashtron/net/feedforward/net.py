from net.feedforward.feedforward_network import FeedforwardNetwork
from net.feedforward.io import FeedforwardNetworkIO
from layer.layer import Layer

class Net:
    def __init__(self):
        self.network = FeedforwardNetwork(self)
        self.io = FeedforwardNetworkIO(self)
    
    @staticmethod
    def new():
        return Net()

    def new_layer(self, n: int, bits: int, premodulo: int = 0) -> None:
        self.network.new_layer(n, bits, premodulo)

    def new_combiner(self, layer: Layer) -> None:
        self.network.new_combiner(layer)
