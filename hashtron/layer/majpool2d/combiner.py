class MajPool2D:
    def __init__(self, vec, width, height, subwidth, subheight, capwidth, capheight, repeat, bias):
        """
        Initialize the MajPool2D combiner.

        :param vec: The vector of boolean values.
        :param width: The width of the pooling matrix.
        :param height: The height of the pooling matrix.
        :param subwidth: The width of the submatrix.
        :param subheight: The height of the submatrix.
        :param capwidth: The width of the capture area.
        :param capheight: The height of the capture area.
        :param repeat: The number of repetitions.
        :param bias: The bias value for majority pooling.
        """
        self.vec = vec
        self.width = width
        self.height = height
        self.subwidth = subwidth
        self.subheight = subheight
        self.capwidth = capwidth
        self.capheight = capheight
        self.repeat = repeat
        self.bias = bias

    def put(self, n: int, v: bool) -> None:
        """
        Set the n-th boolean value in the vector.

        :param n: The index of the boolean value to set.
        :param v: The boolean value to set.
        """
        self.vec[n] = v

    def disregard(self, n: int) -> bool:
        """
        Check if setting the n-th boolean value to False would not affect any feature output.

        :param n: The index of the boolean value to check.
        :return: True if setting the value to False would not affect the output, False otherwise.
        """
        orign = n
        submatrix = self.subwidth * self.subheight
        matrix = self.width * self.height * submatrix
        base = (n // matrix) * matrix
        n %= matrix
        n //= submatrix
        n *= submatrix
        w0, w1 = 0, 0

        for m in range(submatrix):
            if orign == base + m + n:
                w0 += 1
                w1 -= 1
                continue
            if self.vec[base + m + n]:
                w0 += 1
                w1 += 1
            else:
                w0 -= 1
                w1 -= 1

        cond1 = (w0 > self.bias) == (w1 > self.bias)
        return cond1

    def feature(self, m: int) -> int:
        """
        Compute the m-th feature from the combiner.

        :param m: The index of the feature to compute.
        :return: The computed feature as an integer.
        """
        supermatrix = self.width * self.height
        submatrix = self.subwidth * self.subheight
        matrix = supermatrix * submatrix
        base = (m // matrix) * matrix
        starty = m // self.width
        startx = m % self.width
        o = 0

        for y in range(self.capheight):
            for x in range(self.capwidth):
                xx = (x + startx) % self.width
                yy = (y + starty) % self.height
                w = 0

                for m in range(submatrix):
                    if self.vec[base + submatrix * (self.width * yy + xx) + m]:
                        w += 1
                    else:
                        w -= 1

                o <<= 1
                if w > self.bias:
                    o |= 1

        return o
