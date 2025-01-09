class HashtronModel:
    def __init__(self, program=None, bits=1):
        self.program = program if program else []
        self.bits = bits

    def get(self, n: int) -> tuple[int, int]:
        return self.program[n][0], self.program[n][1]

    def len(self) -> int:
        return len(self.program)

    def bits(self) -> int:
        return self.bits

    def set_bits(self, bits: int):
        self.bits = bits
