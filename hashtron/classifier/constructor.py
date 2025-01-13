import random
from hashtron.classifier.model import HashtronModel
from hashtron.classifier.forward import HashtronForward
from hashtron.classifier.view import HashtronView

class Hashtron:
    def __init__(self, program=None, bits=1):
        self.program = program if program else [[random.randint(0, 0xFFFFFFFF) >> 1, 2]]
        self.bits = bits if bits > 0 else 1
        self.model = HashtronModel(self)
        self.forward = HashtronForward(self)
        self.view = HashtronView(self)
    
    @staticmethod
    def new(program=None, bits=1):
        return Hashtron(program, bits)

