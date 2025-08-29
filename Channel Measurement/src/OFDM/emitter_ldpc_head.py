# -*- coding: utf-8 -*-
"""
Emitter with LDPC encoding integrated.
- Reads a TXT file as bits
- Scrambles bits
- LDPC-encodes (IEEE 802.11n/802.16 via ldpc_jossy)
- QPSK maps
- OFDM modulates and plays/plots

Notes:
1) Make sure the ldpc_jossy library is available and (optionally) compiled for decoding.
   Encoding works in pure Python, but you'll still need the package.
   If your ldpc_jossy path is different, adjust LDPC_PY_PATH below.
2) You can tweak LDPC parameters (standard/rate/z/ptype) via the constants near the top.
"""

import struct, zlib
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import chirp, butter, filtfilt, lfilter
import matplotlib.pyplot as plt
import os
import sys

# ---------------------- LDPC library import ----------------------
# If your ldpc_jossy repo path differs, adjust here:
LDPC_PY_PATH = r'D:\\Pycharm\\SEU-CAM-25-Newton-s-Apple\\ldpc_jossy\\py'
if LDPC_PY_PATH and LDPC_PY_PATH not in sys.path:
    sys.path.append(LDPC_PY_PATH)

try:
    import ldpc  # from ldpc_jossy/py
except Exception as e:
    raise ImportError(
        f"无法导入 ldpc 包：{e}\\n"
        f"请检查 LDPC_PY_PATH 是否指向 ldpc_jossy/py 目录，并且包含 __init__.py。"
    )

# ---------------------- Local utils import ----------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils import scrambler,ldpc_encode_bits

# ---------------------- Parameters ----------------------
fs = 48000
N_fft = 4096
cp_len = 1024
num_symbols = 8
# -------- File Header config  --------
HEADER_MAGIC = b'FHDR'
HEADER_VERSION = 1
HEADER_MAX_NAME = 64     # 文件名最多编码 64 字节
HEADER_REP = 3           # 头部LDPC后重复次数（奇数便于多数表决）

# LDPC parameters (defaults are safe; change to your needs)
LDPC_STANDARD = '802.11n'   # '802.11n' or '802.16'
LDPC_RATE = '1/2'           # '1/2','2/3','3/4','5/6'
LDPC_Z = 27                 # for 802.11n usually 27/54/81
LDPC_PTYPE = 'A'            # only used for 802.16 rate 2/3 or 3/4

# ---------------------- I/O paths (adjust if needed) ----------------------
TXT_INPUT_PATH = r"D:\\55495\\个人文件\\剑桥\\OFDM\\shakespace(short).txt"
PILOT_SAVE_PATH = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\save\pilot\pilot_different_N4096_txt_head.npy"
DATA_WAVEFORM_SAVE_PATH = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\save\signal\txt_time_N4096_head.npy"
WAV_SAVE_PATH =  r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\save\signal\tx_signal_N4096_head.wav"

# ---------------------- Helpers ----------------------
def get_bits_from_txt(file_pth: str):
    assert os.path.exists(file_pth), f"file not exist, given arg {file_pth}"
    with open(file_pth, 'rb') as f:
        byte_data = f.read()
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array)
    return bit_array.flatten()

def generate_chirp(fs, duration=2, f0=10, f1=24000):
    t = np.linspace(0, duration, int(fs * duration))
    return chirp(t, f0=f0, f1=f1, t1=duration, method='linear')

def OFDM_modulate_data(symbols, N, cp_len):
    num_symbols = len(symbols) // (N // 2 - 1)
    symbols = symbols[:num_symbols * (N // 2 - 1)]
    data_matrix = symbols.reshape((num_symbols, N // 2 - 1))

    freq_data = np.ones((num_symbols, N), dtype=complex)
    freq_data[:, 1:N // 2] = data_matrix
    freq_data[:, N // 2 + 1:] = np.conj(data_matrix)[:, ::-1]  # Hermitian symmetry
    time_data = np.fft.ifft(freq_data, axis=1)
    cp = time_data[:, -cp_len:]
    with_cp = np.hstack([cp, time_data]).flatten()
    with_cp /= np.max(np.abs(with_cp))
    return np.real(with_cp)

def QPSK_mapping(bits):
    bits = bits.reshape((-1, 2))
    mapping_table = {(0,0): 1+1j, (0,1): -1+1j, (1,0): 1-1j, (1,1): -1-1j}
    syms = np.array([mapping_table[tuple(b)] for b in bits])
    return syms / np.sqrt(2)

def generate_pilot_symbol(N):
    half = N // 2
    real_parts = np.random.choice([-1, 1], size=half - 1)
    imag_parts = np.random.choice([-1, 1], size=half - 1)
    X_half = (real_parts + 1j * imag_parts) / np.sqrt(2)

    X_freq = np.zeros(N, dtype=complex)
    X_freq[0] = 1
    X_freq[1:half] = X_half
    X_freq[half] = 1
    X_freq[half + 1:] = np.conj(X_half[::-1])
    return X_freq

def ofdm_modulate(symbol_freq):
    time_signal = np.fft.ifft(symbol_freq)
    return np.concatenate([time_signal[-cp_len:], time_signal])

# 新增：将字节流转比特
def bytes_to_bits(b: bytes):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)

# 新增：打包文件头（魔数|版本|name_len|size|name|crc32）
def pack_header_bytes(file_path: str) -> bytes:
    name = os.path.basename(file_path)[:HEADER_MAX_NAME]
    name_b = name.encode('utf-8')
    name_len = len(name_b)
    fsize = os.path.getsize(file_path)

    buf = bytearray()
    buf += HEADER_MAGIC                       # 4B
    buf += struct.pack('>B', HEADER_VERSION)  # 1B
    buf += struct.pack('>B', name_len)        # 1B
    buf += struct.pack('>I', fsize)           # 4B, BE
    buf += name_b                             # name_len B
    crc = zlib.crc32(buf) & 0xFFFFFFFF
    buf += struct.pack('>I', crc)             # 4B

    return bytes(buf)

# 按 OFDM 子载波容量在“比特域”补齐；使用 PRBS7 随机填充，避免模式化
def pad_bits_for_ofdm(bits: np.ndarray, N_fft: int, prbs_seed: int = 0x6D) -> np.ndarray:
    bits = (bits.astype(np.uint8) & 1)
    per_qpsk = N_fft // 2 - 1            # 每个 OFDM 符号可承载的 QPSK 点数
    bits_per_ofdm = 2 * per_qpsk         # QPSK 每点 2 比特
    # 需要补齐到整数个 OFDM 符号
    need = (-len(bits)) % bits_per_ofdm
    if len(bits) == 0:
        need = bits_per_ofdm

    if need:
        # PRBS7: 多项式 x^7 + x^6 + 1
        state = prbs_seed & 0x7F or 0x5D  # 不能为 0
        filler = np.empty(need, dtype=np.uint8)
        for i in range(need):
            newbit = ((state >> 6) ^ (state >> 5)) & 1
            state = ((state << 1) & 0x7F) | newbit
            filler[i] = newbit
        bits = np.concatenate([bits, filler])
    return bits

# ---------------------- Main flow ----------------------
def main():
    # Pilots
    chirp_sig = generate_chirp(fs, f0=10, f1=24000)
    chirp_tail = generate_chirp(fs, f0=20, f1=18000)

    tx_signal = np.array([])
    mode = input("【mode】1 for different, 2 for same ：")
    if mode == '1':
        pilot_different = []
        for _ in range(num_symbols):
            pilot = generate_pilot_symbol(N_fft)
            pilot_different.append(pilot)
            ofdm_time = ofdm_modulate(pilot)
            tx_signal = np.concatenate([tx_signal, ofdm_time])
        np.save(PILOT_SAVE_PATH, pilot_different)
    else:
        pilot = generate_pilot_symbol(N_fft)
        ofdm_time = ofdm_modulate(pilot)
        np.save(PILOT_SAVE_PATH, pilot)
        for _ in range(num_symbols):
            tx_signal = np.concatenate([tx_signal, ofdm_time])

    tx_signal_real = np.real(tx_signal)
    tx_signal_real /= np.max(np.abs(tx_signal_real))
    ch = input("【chirp】1 for two chirp , 2 for one chirp : ")
    if ch == '1':
        tx_signal_real = np.concatenate([chirp_sig, tx_signal_real, chirp_tail])
    else:
        tx_signal_real = np.concatenate([chirp_sig, tx_signal_real])

    # ---- 在文本内容前插入：强保护文件头（修正后的顺序） ----
    hdr_bytes = pack_header_bytes(TXT_INPUT_PATH)

    # 1) 字节 -> 比特（uint8）
    hdr_bits = bytes_to_bits(hdr_bytes).astype(np.uint8)

    # 2) 扰码（要求 uint8；若你的 scrambler 内部未做转换，建议其入口也 cast 一次）
    hdr_bits_scr = scrambler(hdr_bits).astype(np.uint8)
    # hdr_bits_scr = hdr_bits

    # 3) LDPC 编码 + 重复（保持 uint8）
    hdr_ldpc_bits, _ = ldpc_encode_bits(hdr_bits_scr)
    hdr_ldpc_bits_rep = np.tile(hdr_ldpc_bits.astype(np.uint8), HEADER_REP)

    # 4) ★在比特域按 OFDM 容量补齐（避免出现复数/浮点进 XOR）
    hdr_ldpc_bits_rep = pad_bits_for_ofdm(hdr_ldpc_bits_rep, N_fft)

    # 5) QPSK 映射 -> OFDM 调制
    hdr_syms = QPSK_mapping(hdr_ldpc_bits_rep)  # 输入已保证偶数且整 OFDM 长度
    header_waveform = OFDM_modulate_data(hdr_syms, N_fft, cp_len)

    # 读取 & 扰码
    bits = get_bits_from_txt(TXT_INPUT_PATH)
    print(f"raw bits: {bits.shape}")
    bits_scr = scrambler(bits).astype(np.uint8)
    # bits_scr = bits.astype(np.uint8)
    print(f"scrambled bits (first 16): {bits_scr[:16]}")

    # -------- LDPC 编码 --------
    ldpc_bits, (K, Ncw) = ldpc_encode_bits(bits_scr)
    print(f"LDPC params: K={K}, N={Ncw}, coded length={len(ldpc_bits)}")

    # 交调制（QPSK）
    # 确保偶数长度
    if len(ldpc_bits) % 2 == 1:
        ldpc_bits = np.concatenate([ldpc_bits, np.array([0], dtype=np.uint8)])
    qpsk_symbols = QPSK_mapping(ldpc_bits)
    print(f"QPSK symbols: {qpsk_symbols.shape}")

    # OFDM 调制并保存
    data_waveform = OFDM_modulate_data(qpsk_symbols, N_fft, cp_len)
    np.save(DATA_WAVEFORM_SAVE_PATH, np.array(data_waveform))

    # 合成最终发送波形：导频/前导 + 文件头 + 文本数据
    # tx_signal_realtime = np.concatenate([tx_signal_real, header_waveform, data_waveform])
    tx_signal_realtime = np.concatenate([tx_signal_real, header_waveform, data_waveform , chirp_tail])

    # 归一化后播放
    # signal_cut = tx_signal_realtime.copy()
    # signal_cut /= np.max(np.abs(signal_cut))
    # signal_cut = np.concatenate([signal_cut, chirp_tail])
    signal_cut = tx_signal_realtime
    np.save(r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\save\signal\wholesignal_4096.npy", signal_cut)
    print("🔊 Playing the transmit signal...")
    # sd.play(signal_cut, fs)
    # sd.wait()
    # np.save(r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\record\whole_signal.npy",signal_cut)

    plt.figure(figsize=(12, 6))
    plt.plot(tx_signal_realtime, label='TX (LDPC-coded)')
    plt.title("Transmit Signal with LDPC-coded payload")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(100000, 200000), tx_signal_realtime[100000:200000], label='TX (LDPC-coded)')
    plt.title("Transmit Signal with LDPC-coded payload (100000-200000 samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    if WAV_SAVE_PATH:
        write(WAV_SAVE_PATH, fs, (signal_cut * 32767).astype(np.int16))

    print("✅ Transmission (LDPC-coded) completed")

if __name__ == "__main__":
    main()
