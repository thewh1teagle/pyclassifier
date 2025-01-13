class SingleValue:
    def __init__(self, num):
        self.num = num

    def feature(self, n: int) -> int:
        return self.num

    def parity(self) -> int:
        return 0
