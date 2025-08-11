import os
import numpy as np

def scrambler(bits, seed=0b1111111):
    """
    Simple LFSR-based scrambler to randomize bit sequence.
    :param bits: input bit array (0s and 1s)
    :param seed: initial state of LFSR (7-bit)
    :return: scrambled bit array
    """
    state = seed
    out = np.empty_like(bits)
    for i in range(len(bits)):
        newbit = ((state >> 6) ^ (state >> 3)) & 1
        out[i] = bits[i] ^ newbit
        state = ((state << 1) & 0x7f) | newbit
    return out