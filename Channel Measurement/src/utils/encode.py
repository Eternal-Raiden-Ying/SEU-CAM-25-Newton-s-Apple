import os
import numpy as np


def scrambler(bits, seed=0b1111111, bit_width=7):
    """
        Simple LFSR-based scrambler to randomize bit sequence.

        v2: add parameter bit_width, so you can choose different types of scrambler, supported [5,6,7,9,10,11]
            to extend, see more about primitive polynomial
    :param bits: input bit array (0s and 1s)
    :param seed: initial state of LFSR (7-bit)
    :param bit_width: bit width of LFSR
    :return: scrambled bit array
    """
    state = seed
    out = np.empty_like(bits)
    if bit_width == 5:
        a, b = 4, 1
        c = 0x1f
    elif bit_width == 6:
        a, b = 5, 0
        c = 0x3f
    elif bit_width == 7:
        a, b = 6, 3
        c = 0x7f
    elif bit_width == 9:
        a, b = 8, 4
        c = 0x1ff
    elif bit_width == 10:
        a, b = 9, 6
        c = 0x3ff
    elif bit_width == 11:
        a, b = 10, 8
        c = 0x7ff
    else:
        raise ValueError(f"Not supported bit width for {bit_width}, see more details at scrambler()")

    for i in range(len(bits)):
        newbit = ((state >> a) ^ (state >> b)) & 1
        out[i] = bits[i] ^ newbit
        state = ((state << 1) & c) | newbit
        # if you want to see how this function work, run the code below
        print(f"i:{i}, newbit:{newbit}, ", f"new_state:{state:>7b}".replace(' ', '0'))
    return out


if __name__ == "__main__":
    bits = np.random.randint(0, 2, 127)
    bits_s = scrambler(bits)
    # bits_ds = scrambler(bits)
    print()
