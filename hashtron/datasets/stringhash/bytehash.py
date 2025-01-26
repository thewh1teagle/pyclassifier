from hashtron.hash.hash import Hash

class ByteSample:
    def __init__(self, buf: bytes, out: int):
        self.buf = buf
        self.out = out

    def feature(self, n: int) -> int:
        ret = 0
        for j in range(4):
            index = Hash.hash(n, j, len(self.buf))
            byte_val = self.buf[index]
            ret ^= (byte_val << (8 * j))
        return ret

    def parity(self) -> int:
        return 0

    def output(self) -> int:
        return self.out

class BalancedByteSample:
    def __init__(self, buf: bytes, out: int):
        self.buf = buf
        self.out = out

    def feature(self, n: int) -> int:
        ret = 0
        for j in range(4):
            index = Hash.hash(n, j, len(self.buf))
            byte_val = self.buf[index]
            ret ^= (byte_val << (8 * j))
        return ret

    def parity(self) -> int:
        ret = 0
        for b in self.buf:
            ret = Hash.hash(ret, b, 0xFFFFFFFF)
        return ret & 0xFFFF

    def output(self) -> int:
        return self.out
