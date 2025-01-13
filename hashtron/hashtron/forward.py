from hash.hash import Hash

class HashtronForward:
    def __init__(self, hashtron):
        self.hashtron = hashtron

    def forward(self, sample: int, negate: bool) -> int:
        if not self.hashtron.program:
            return 0
        out = 0
        for j in range(self.hashtron.bits):
            input_val = sample if self.hashtron.bits <= 1 else sample | (j << 16)
            ss, maxx = self.hashtron.program[0]
            input_val = Hash.hash(input_val, ss, maxx)
            for i in range(1, len(self.hashtron.program)):
                s, max_val = self.hashtron.program[i]
                ss ^= s
                maxx -= max_val
                input_val = Hash.hash(input_val, ss, maxx)
            input_val &= 1
            if negate:
                input_val ^= 1
            if input_val != 0:
                out |= 1 << j
        return out
