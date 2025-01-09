class Hash:
    @staticmethod
    def hash(n: int, s: int, max_val: int) -> int:
        # Mixing stage, mix input with salt using subtraction
        m = (n - s) & 0xFFFFFFFF
        # Hashing stage, use xor shift with prime coefficients
        m ^= (m << 2) & 0xFFFFFFFF
        m ^= (m << 3) & 0xFFFFFFFF
        m ^= (m >> 5) & 0xFFFFFFFF
        m ^= (m >> 7) & 0xFFFFFFFF
        m ^= (m << 11) & 0xFFFFFFFF
        m ^= (m << 13) & 0xFFFFFFFF
        m ^= (m >> 17) & 0xFFFFFFFF
        m ^= (m << 19) & 0xFFFFFFFF
        # Mixing stage 2, mix input with salt using addition
        m += s
        m &= 0xFFFFFFFF
        # Modular stage using Lemire's fast alternative to modulo reduction
        return ((m * max_val) >> 32) & 0xFFFFFFFF

    @staticmethod
    def strings_hash(in_val: int, strs: list[str]) -> int:
        out = in_val
        for s in strs:
            out = Hash.string_hash(out, s)
        return out

    @staticmethod
    def string_hash(in_val: int, s: str) -> int:
        out = in_val
        for c in s:
            out = Hash.hash(out, ord(c), 0xFFFFFFFF)
        return out
