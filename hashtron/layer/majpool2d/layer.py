from hashtron.layer.majpool2d.combiner import MajPool2D
from hashtron.layer.layer import Layer

class MajPool2DLayer(Layer):
    def __init__(self, width, height, subwidth, subheight, capwidth, capheight, repeat, bias=0):
        self.width = width
        self.height = height
        self.subwidth = subwidth
        self.subheight = subheight
        self.capwidth = capwidth
        self.capheight = capheight
        self.repeat = repeat
        self.bias = bias

    def lay(self) -> MajPool2D:
        vec = [False] * (self.width * self.height * self.subwidth * self.subheight * self.repeat)
        return MajPool2D(vec, self.width, self.height, self.subwidth, self.subheight, self.capwidth, self.capheight, self.repeat, self.bias)
