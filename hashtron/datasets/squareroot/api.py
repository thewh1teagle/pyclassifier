import math
from typing import List

class Sample:
    def __init__(self, value: int):
        self.value = value

    def feature(self, n: int) -> int:
        return self.value

    def parity(self) -> int:
        # don't balance: return 0
        return 4
        # return self.value ^ (self.value << 3)

    def output(self) -> int:
        return int(math.sqrt(self.value))

SmallClasses = 4
MediumClasses = 5
BigClasses = 6
HugeClasses = 7

def small() -> List[Sample]:
    return [Sample(i) for i in range(1 << 8)]

def medium() -> List[Sample]:
    return [Sample(i) for i in range(1 << 10)]

def big() -> List[Sample]:
    return [Sample(i) for i in range(1 << 12)]

def huge() -> List[Sample]:
    return [Sample(i) for i in range(1 << 14)]
