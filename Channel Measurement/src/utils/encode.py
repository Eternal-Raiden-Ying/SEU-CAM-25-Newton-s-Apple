import os
import numpy as np
import sys
LDPC_PY_PATH = r'D:\\Pycharm\\SEU-CAM-25-Newton-s-Apple\\ldpc_jossy\\py'
if LDPC_PY_PATH and LDPC_PY_PATH not in sys.path:
    sys.path.append(LDPC_PY_PATH)
import ldpc

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

def scrambler_random(bits, seed=42):
    """
    使用固定随机种子生成伪随机 bit 流进行按位异或 scrambler
    :param bits: 输入 bit 数组（0/1）
    :param seed: 随机种子
    :return: scrambled bit 数组
    """
    rng = np.random.default_rng(seed)  # 创建随机生成器
    prbs = rng.integers(0, 2, size=len(bits), dtype=np.uint8)  # 生成 0/1 伪随机序列
    scrambled = np.bitwise_xor(bits, prbs)  # 按位异或
    return scrambled.astype(np.uint8)


if __name__ == "__main__":
    bits = np.array([0,0,1,0,1,1,0,1])
    bits_scra = scrambler(bits)
    bits_descra = scrambler(bits_scra)