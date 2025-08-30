import os
import numpy as np
import sys
from typing import Dict, Tuple, Optional
from .ldpc_jossy import code

__all__ = ['ldpc_make_code', 'ldpc_encode_bits', 'scramble_bits']

# LDPC default parameters (defaults are safe; change when invoking the function)
LDPC_STANDARD = '802.11n'   # '802.11n' or '802.16'
LDPC_RATE = '1/2'           # '1/2','2/3','3/4','5/6'
LDPC_Z = 27                 # for 802.11n usually 27/54/81
LDPC_PTYPE = 'A'            # only used for 802.16 rate 2/3 or 3/4

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
        # print(f"i:{i}, newbit:{newbit}, ", f"new_state:{state:>7b}".replace(' ', '0'))
    return out

def ldpc_encode_bits(in_bits,*,
                     c = None,
                     standard=LDPC_STANDARD,
                     rate=LDPC_RATE,
                     z=LDPC_Z,
                     ptype=LDPC_PTYPE):
    """
    Breaks input bits into K-length blocks and LDPC-encodes each block.
    Returns concatenated codeword bits.
    """
    if c is None:
        c = code(standard=standard, rate=rate, z=z, ptype=ptype)
    K, N = c.K, c.N

    # Trim or pad input to multiple of K
    n_full = len(in_bits) // K
    rem = len(in_bits) % K
    if rem != 0:
        pad = K - rem
        in_bits = np.concatenate([in_bits, np.random.randint(0,2,pad,dtype=np.uint8)])
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
    rng = np.random.default_rng(seed) # 创建随机生成器
    prbs = rng.integers(0, 2, size=len(bits), dtype=np.uint8) # 生成 0/1 伪随机序列
    scrambled = np.bitwise_xor(bits, prbs) # 按位异或
    return scrambled.astype(np.uint8)

def ldpc_make_code(*,
                   standard: str,
                   rate: str,
                   z: int,
                   ptype: str,
                   device: str = "cuda",
                   llr_clip: float = 20.0,
                   max_iter: int = 200,
                   verbose: bool = False,
                   log_every: int = 1,
                   check_every: int = 1,
                   microbatch: int = 256,
                   print_iter: bool = False):
    """
    实例化 LDPC code，并把解码超参配置到实例属性上（供 code.decode 使用）。
    这些属性名与你给的 ldpc.py 保持一致：
      c.dgl_device, c.dgl_llr_clip, c.dgl_max_iter, c.dgl_verbose, c.dgl_log_every, c.dgl_check_every
    """
    c = code(standard=standard, rate=rate, z=z, ptype=ptype)

    # —— 按原版要求写入实例属性（供 decode 使用）——
    c.dgl_device       = device           # e.g. 'cuda' / 'cuda:0' / 'cpu'
    c.dgl_llr_clip     = float(llr_clip)
    c.dgl_max_iter     = int(max_iter)
    c.dgl_verbose      = bool(verbose)
    c.dgl_log_every    = int(log_every)
    c.dgl_check_every  = int(check_every)
    c.dgl_microbatch   = int(microbatch)
    c.print_iter       = bool(print_iter)

    return c


def scramble_bits(bits: np.ndarray, *, seed: int, mode: str = 'random', bit_width: int | None = None) -> np.ndarray:
    if mode == 'random':
        return scrambler_random(bits.astype(np.uint8), seed=seed)
    elif mode == 'LFSR':
        assert isinstance(bit_width, int)
        return scrambler(bits.astype(np.uint8), seed=seed, bit_width=bit_width)
    else:
        raise ValueError(f"Unknown mode {mode}")


if __name__ == "__main__":
    bits = np.array([0,0,1,0,1,1,0,1])
    bits_scra = scrambler(bits)
    bits_descra = scrambler(bits_scra)