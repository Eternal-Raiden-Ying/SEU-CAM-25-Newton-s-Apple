import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft

# ---------- 参数 ----------
N_fft = 1024
cp_len = 32
symbol_len = N_fft + cp_len
data_bins = np.arange(1, 512)  # 有效子载波索引

# ---------- 读取音频 ----------
fs, signal = wav.read(r"..\data\file14.wav")  # 替换为你的路径
signal = signal.astype(np.float32)

# ---------- 读取 channel impulse response（CSV）----------
h = pd.read_csv(r"..\data\channel.csv", header=None).values.flatten()
H = fft(h, N_fft)[1:512]  # 提取有效子载波范围的频域响应

# ---------- 切分 OFDM 符号 & 去除循环前缀 ----------
n_symbols = len(signal) // symbol_len
ofdm_symbols = np.array([
    signal[i * symbol_len + cp_len : (i + 1) * symbol_len]
    for i in range(n_symbols)
])

# ---------- FFT ----------
freq_data = fft(ofdm_symbols, axis=1)
data_subcarriers = freq_data[:, data_bins]
print(type(data_subcarriers))
print(data_subcarriers.shape)
# ---------- 频域均衡（Zeroforcing）----------
equalized = data_subcarriers / H

# ---------- QPSK Gray 解调 ----------
def qpsk_demod(symbols):
    bits = []
    for s in symbols.flatten():
        angle = np.angle(s)
        if angle < 0:
            angle += 2 * np.pi
        if 0 <= angle < np.pi / 2:
            bits.extend([0, 0])
        elif np.pi / 2 <= angle < np.pi:
            bits.extend([0, 1])
        elif np.pi <= angle < 3 * np.pi / 2:
            bits.extend([1, 1])
        elif 3 * np.pi /2 <= angle < np.pi * 2:
            bits.extend([1, 0])
        else:
            raise ValueError(f"unexpected phase for {angle}")
    return np.array(bits, dtype=np.uint8)

bits = qpsk_demod(equalized)

# ---------- 比特转字节 ----------
def bits_to_bytes(bits):
    bits = bits[:len(bits) // 8 * 8] #  TODO: 871*511/8 =xx.125
    return np.packbits(bits)

byte_data = bits_to_bytes(bits)

# ---------- 文件头解析 ----------
all_bytes = byte_data.tobytes()
parts = all_bytes.split(b'\x00')
filename = parts[0].decode(errors='ignore')
filesize = int(parts[1].decode(errors='ignore'))
header_length = len(parts[0]) + 1 + len(parts[1]) + 1

# ---------- 提取文件数据 ----------
file_bytes = byte_data[header_length : header_length + filesize]

# ---------- 确保目录存在 ----------
if os.path.dirname(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

# ---------- 保存文件 ----------
with open(filename, "wb") as f:
    f.write(file_bytes)

print("✅ 解调成功，保存文件为：", filename)


# below are the code and result to extract the header of file14(broken)
# "".join(list(map(lambda x: hex(x)[-2:], byte_data[header_length : header_length+44]))).replace('x','0')
# file14: '524946469440090057415645666d74201000000201000100401f0000803e0000020010006461746170400900'
# file04: '52494646d4f3010057415645666d74201000000001000100401f0000803e00000200100064617461b0f30100'
