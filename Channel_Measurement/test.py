import os
import numpy as np

from module.utils.modulate import QPSK_mapping,OFDM_modulate


def generate_8190bit_probe():
    """ 生成8190比特CAZAC-QPSK信道探测序列 """
    total_bits = 8190
    n_symbols = total_bits // 2
    r = 17  # 与4095互质的根索引

    # 生成CAZAC序列
    n = np.arange(n_symbols)
    cazac = np.exp(-1j * np.pi * r * n * (n + 1) / n_symbols)

    # QPSK量化
    bits = np.zeros(total_bits, dtype=int)
    for i in range(n_symbols):
        bits[2 * i + 1] = 0 if np.real(cazac[i]) >= 0 else 1
        bits[2 * i ] = 0 if np.imag(cazac[i]) >= 0 else 1
    print ('df', len (bits))
    return bits



file_pth = r"D:\Documents\Coding\Python\SEUCAM\Channel_Measurement\record\noise_original.bin"
with open(file_pth, 'rb') as file:
    data = file.read()


bits = generate_8190bit_probe()

const = QPSK_mapping(bits.reshape(-1,2))
pilot = OFDM_modulate(const, N=8192, cp_len=1024)
print(pilot)