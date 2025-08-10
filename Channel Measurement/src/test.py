import os
import sys
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import chirp


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import (generate_chirp, QPSK_mapping, OFDM_modulate, normalize,
                   get_bits_from_file, serial_to_parallel, get_bits_from_str, random_bits)

# 参数设置
fs = 48000  # 采样率
N = 4096    # OFDM 子载波数
cp_len = 128  # 循环前缀长度

chirp_len = 2
chirp_fl = 10
chirp_fh = 24000

if __name__ == "__main__":
    txt_path = r"D:\55495\个人文件\剑桥\OFDM\shakespace_poem.txt"

    bits = get_bits_from_file(txt_path)
    bits_parallel = serial_to_parallel(bits, N=N)
    constellations = QPSK_mapping(bits_parallel)
    symbols = OFDM_modulate(constellations=constellations,
                            N=N, cp_len=cp_len, complement_val=0)

    symbols = normalize(symbols).flatten()
    chirp = generate_chirp(fs=fs, duration=chirp_len, f_l=chirp_fl, f_h=chirp_fh)

    tx = np.concatenate([chirp*0.3, np.real(symbols)])
    sd.play(tx)
    sd.wait()