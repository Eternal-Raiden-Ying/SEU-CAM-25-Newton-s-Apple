import sys
import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import correlate, chirp

from utils import draw_constellation_map, QPSK_reflection

#%%
# 系统参数
fs = 48000
N = 4096
K = 24
pilot_indices = np.arange(1, N//2, K)  # 导频子载波索引: 0, 6, 12, ...
data_indices = np.setdiff1d(np.arange(1,N//2), pilot_indices)
cp_length = 128
tx = np.load(r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\shakespare\pilot.npy")

tx_bits = QPSK_reflection(tx[:,data_indices]).flatten()
def generate_chirp(fs, duration=1):
    t = np.linspace(0, duration, int(fs * duration))
    chirp_sig = chirp(t, f0=10, f1=24000, t1=duration, method='linear')
    return chirp_sig

# Hamming码解码
def hamming_decode(codeword):
    # 输入：38位Hamming码，输出：32位数据
    m = 6
    n = 38
    syndrome = 0
    for i in range(m):
        check_pos = 2**i
        parity = 0
        for j in range(n):
            if (j + 1) & check_pos:
                parity ^= codeword[j]
        syndrome |= (parity << i)
    if syndrome != 0 and syndrome <= n:
        codeword[syndrome - 1] ^= 1  # 纠正错误
    # 提取数据位
    data_idx = [i for i in range(n) if i not in [0, 1, 2, 4, 8, 16, 32]]
    data_bits = codeword[data_idx]
    return data_bits

def time_sync(rx_signal, chirp_sig):
    correlation = np.abs(correlate(rx_signal, chirp_sig, mode='full'))
    start_idx = np.argmax(correlation)
    print(f"Correlation peak: {np.max(correlation)}, Start index: {start_idx}")
    return start_idx

def remove_cyclic_prefix(ofdm_time):
    return ofdm_time[cp_length:cp_length + N]

def generate_zadoff_chu(length, u=1):
    n = np.arange(length)
    return np.exp(-1j * np.pi * u * n * (n + 1) / length)

def channel_estimation(rx_freq, pilot_symbols):
    H_est = np.zeros(N, dtype=complex)
    H_pilot = rx_freq[pilot_indices] / pilot_symbols
    for i in range(N):
        if i in pilot_indices:
            H_est[i] = H_pilot[np.where(pilot_indices == i)[0][0]]
        else:
            left_idx = pilot_indices[np.where(pilot_indices < i)[0][-1]] if np.any(pilot_indices < i) else pilot_indices[0]
            right_idx = pilot_indices[np.where(pilot_indices > i)[0][0]] if np.any(pilot_indices > i) else pilot_indices[-1]
            left_val = H_pilot[np.where(pilot_indices == left_idx)[0][0]]
            right_val = H_pilot[np.where(pilot_indices == right_idx)[0][0]]
            weight = (i - left_idx) / (right_idx - left_idx)
            H_est[i] = left_val + (right_val - left_val) * weight
    return H_est

def bits_to_bytes(bits):
    bits = bits[:len(bits) // 8 * 8]
    return np.packbits(bits)

def receiver():
    rx_signal = np.load(r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\shakespare\received_signal_shakeapace_2.npy")
    chirp_sig = generate_chirp(fs)
    start_idx = time_sync(rx_signal, chirp_sig)
    rx_signal = rx_signal[start_idx + 1:]

    pilot_symbols = generate_zadoff_chu(len(pilot_indices))
    symbol_length = N + cp_length
    num_symbols = len(rx_signal) // symbol_length
    rx_symbols = np.zeros((num_symbols, N), dtype=complex)

    for i in range(num_symbols):
        ofdm_time = rx_signal[i * symbol_length:(i + 1) * symbol_length]
        if len(ofdm_time) < symbol_length:
            break
        ofdm_time_no_cp = remove_cyclic_prefix(ofdm_time)
        rx_symbols[i] = np.fft.fft(ofdm_time_no_cp, N)

    data_symbols = []
    for i in range(num_symbols):
        rx_freq = rx_symbols[i]
        H_est = channel_estimation(rx_freq, pilot_symbols)
        eq_symbols = rx_freq[data_indices] / H_est[data_indices]
        # if i % 50 ==0:
        #     draw_constellation_map(eq_symbols, emit_pilot=tx[i][data_indices])
        #     plt.show()

        data_symbols.extend(eq_symbols)
    bits = QPSK_reflection(np.array(data_symbols)).flatten()

    # 文件头解码
    filename_bits = bits[:304]  # 8块Hamming(38,32)
    filename_decoded = []
    for i in range(0, 304, 38):
        block = filename_bits[i:i+38]
        filename_decoded.append(hamming_decode(block))
    filename_bits = np.concatenate(filename_decoded)  # 256位
    filesize_bits = hamming_decode(bits[304 + 8:304 + 8 + 38])  # 跳过分隔符
    filesize = int(''.join(str(b) for b in filesize_bits), 2)

    # 解析文件头
    byte_data = bits_to_bytes(bits[304 + 8 + 38 + 8:])  # 跳过文件头（304+8+38+8=358位）
    all_bytes = byte_data.tobytes()
    filename = np.frombuffer(filename_bits.tobytes(), dtype=np.uint8).tobytes().decode('utf-8', errors='ignore').rstrip('\x00')

    # 提取文件数据
    file_bytes = byte_data[:filesize]

    # 保存文件
    output_path = r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\output\OFDM\\" + filename
    with open(output_path, 'wb') as f:
        f.write(file_bytes.tobytes())
    print(f"✅ Received file saved to {output_path}")

    plt.plot(rx_signal)
    plt.title("接收信号")
    plt.xlabel("采样点")
    plt.ylabel("幅度")
    plt.grid(True)
    plt.show()

    return filename, filesize
#%%
# 示例调用
if __name__ == "__main__":
    filename, filesize = receiver()
    print(f"Received file: {filename}, Size: {filesize} bytes")