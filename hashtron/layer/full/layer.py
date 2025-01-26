from .combiner import Full

class FullLayer:
    def __init__(self, size: int, bits: int, maxbits: int):
        """
        Fully connected layer configuration.
        
        Args:
            size (int): Total size of the boolean vector.
            bits (int): Number of bits per feature.
            maxbits (int): Maximum bits to read per feature.
        """
        self.size = size
        self.bits = bits
        self.maxbits = maxbits

    def lay(self) -> Full:
        """Create a Full combiner instance with initialized boolean vector."""
        vec = [False] * self.size
        return Full(vec, self.bits, self.maxbits)
