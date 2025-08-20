import os
import numpy as np
import sys
LDPC_PY_PATH = r'D:\\Pycharm\\SEU-CAM-25-Newton-s-Apple\\ldpc_jossy\\py'
if LDPC_PY_PATH and LDPC_PY_PATH not in sys.path:
    sys.path.append(LDPC_PY_PATH)
import ldpc


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

# LDPC parameters (defaults are safe; change to your needs)
LDPC_STANDARD = '802.11n'   # '802.11n' or '802.16'
LDPC_RATE = '1/2'           # '1/2','2/3','3/4','5/6'
LDPC_Z = 27                 # for 802.11n usually 27/54/81
LDPC_PTYPE = 'A'            # only used for 802.16 rate 2/3 or 3/4

def ldpc_encode_bits(in_bits,
                     standard=LDPC_STANDARD,
                     rate=LDPC_RATE,
                     z=LDPC_Z,
                     ptype=LDPC_PTYPE):
    """
    Breaks input bits into K-length blocks and LDPC-encodes each block.
    Returns concatenated codeword bits.
    """
    c = ldpc.code(standard=standard, rate=rate, z=z, ptype=ptype)
    K, N = c.K, c.N

    # Trim or pad input to multiple of K
    n_full = len(in_bits) // K
    rem = len(in_bits) % K
    if rem != 0:
        pad = K - rem
        in_bits = np.concatenate([in_bits, np.zeros(pad, dtype=np.uint8)])
        n_full += 1

    in_bits = in_bits.reshape(n_full, K)
    codewords = []
    for u in in_bits:
        x = c.encode(u.astype(int))  # returns length N, {0,1}
        codewords.append(np.array(x, dtype=np.uint8))

    cw = np.concatenate(codewords)
    return cw, (K, N)

if __name__ == "__main__":
    bits = np.array([0,0,1,0,1,1,0,1])
    bits_scra = scrambler(bits)
    bits_descra = scrambler(bits_scra)