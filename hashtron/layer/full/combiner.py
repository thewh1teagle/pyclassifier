from typing import List

class Full:
    def __init__(self, vec: List[bool], bits: int, maxbits: int):
        """
        Combiner for the fully connected layer.
        
        Args:
            vec (List[bool]): Boolean vector storing layer state.
            bits (int): Number of bits per feature.
            maxbits (int): Maximum bits to read for a feature.
        """
        self.vec = vec
        self.bits = bits
        self.maxbits = maxbits

    def put(self, n: int, v: bool) -> None:
        """Set the n-th boolean value in the vector."""
        self.vec[n] = v

    def disregard(self, n: int) -> bool:
        """Check if setting the n-th value to False has no effect (always False here)."""
        return False

    def feature(self, m: int) -> int:
        """
        Compute the m-th feature by packing `maxbits` bits from the vector.
        
        Args:
            m (int): Feature index.
            
        Returns:
            int: Packed integer formed from the boolean vector bits.
        """
        start = m * self.bits
        end = start + self.maxbits
        
        # Replicate Go's out-of-bounds check
        if end > len(self.vec):
            return 0
        
        o = 0
        for pos in range(start, end):
            o <<= 1
            if self.vec[pos]:
                o |= 1
        
        # Truncate to 32-bit unsigned integer (Go's uint32)
        return o & 0xFFFFFFFF
