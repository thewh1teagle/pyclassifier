from hashtron.hash.hash import Hash

class Sample:
    def __init__(self, hash: str):
        self.hash = hash

    def feature(self, n: int) -> int:
        return Hash.string_hash(n, self.hash)

    def parity(self) -> int:
        return 0


class BalancedSample:
    def __init__(self, hash: str):
        self.hash = hash

    def feature(self, n: int) -> int:
        return Hash.string_hash(n, self.hash)

    def parity(self) -> int:
        return Hash.string_hash(0xffffffff, self.hash)


