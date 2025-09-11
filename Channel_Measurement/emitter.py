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
from __future__ import annotations
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import chirp, butter, filtfilt, lfilter
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
from PIL import Image
import os


# ---------- LDPC library import and params (match TX) ----------
LDPC_PY_PATH = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel_Measurement\module\utils\ldpc_jossy\py"
if LDPC_PY_PATH and LDPC_PY_PATH not in sys.path:
    sys.path.append(LDPC_PY_PATH)

try:
    import ldpc  # from ldpc_jossy/py
except Exception as e:
    raise ImportError(
        f"无法导入 ldpc 包：{e}\\n"
        f"请检查 LDPC_PY_PATH 是否指向 ldpc_jossy/py 目录，并且包含 __init__.py。"
    )

# ---------------------- Parameters ----------------------
fs = 48000
N_fft = 8192
cp_len = 1024
num_symbols = 8

# LDPC parameters (defaults are safe; change to your needs)
LDPC_STANDARD = '802.11n'  # '802.11n' or '802.16'
LDPC_RATE = '1/2'  # '1/2','2/3','3/4','5/6'
LDPC_Z = 81  # for 802.11n usually 27/54/81
LDPC_PTYPE = 'A'  # only used for 802.16 rate 2/3 or 3/4

# ---------------------- I/O paths (adjust if needed) ----------------------
TXT_INPUT_PATH = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel_Measurement\data\shakespace_poem_middle.txt"
TIFF_INPUT_PATH = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel_Measurement\data\answer.tiff"
PILOT_SAVE_PATH = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel_Measurement\save\pilot\pilot_different_N8192_same.npy"
# DATA_WAVEFORM_SAVE_PATH =r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel_Measurement\save\signal\signal_N8192_tiff_nocomb.npy"
DATA_WAVEFORM_SAVE_PATH =r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel_Measurement\save\signal\signal_N8192_tiff_nocomb.npy"
WAV_SAVE_PATH =r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel_Measurement\save\signal\tx_signal_N8192_nocomb.wav"


# ---------------------- Helpers ----------------------
# 只用正频 90% 子载波承载数据，剩余 10% 用随机 QPSK 填充
RANDOM_FILL_SEED = 2025  # 随机填充的种子；Tx/Rx 不需要共享（Rx会屏蔽这部分）

def prepend_silence_complex(x: np.ndarray, fs: float, silence_sec: float) -> np.ndarray:
    """x 为复数基带信号；在最前面加 silence_sec 秒的全零静音"""
    L = int(round(fs * silence_sec))
    pad = np.zeros(L, dtype=x.dtype if np.iscomplexobj(x) else np.complex64)
    return np.concatenate([pad, x])

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
        c = ldpc.code(standard=standard, rate=rate, z=z, ptype=ptype)
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

# =================== PATCH: 64-bit长度头工具 ===================
def u64_to_bits_msb(n: int) -> np.ndarray:
    """
    把整数 n 转为 64 位（MSB-first，大端）的 bit 数组（dtype=uint8, 值∈{0,1}）
    例：n=5 -> '000...0101'
    """
    n = int(n) & ((1 << 64) - 1)
    bstr = format(n, '064b')  # 64位二进制字符串，MSB在左
    return np.fromiter((1 if c == '1' else 0 for c in bstr), dtype=np.uint8)

def ascii3_to_24bits(s3: str) -> np.ndarray:
    """
    把 3 个 ASCII 字符编码成 24bit（MSB-first）
    例如 'txt' / 'tif' / 'bin'
    """
    assert len(s3) == 3, "file type must be 3 chars"
    val = (ord(s3[0]) << 16) | (ord(s3[1]) << 8) | ord(s3[2])
    b = np.zeros(24, dtype=np.uint8)
    for k in range(24):
        b[k] = (val >> (23 - k)) & 1
    return b

def u40_to_bits_msb(n: int) -> np.ndarray:
    """
    把整数 n 转 40bit（MSB-first）。用于“文件大小（bit数）”。
    """
    assert 0 <= n < (1 << 40), "payload length must fit in 40 bits"
    b = np.zeros(40, dtype=np.uint8)
    for k in range(40):
        b[k] = (n >> (39 - k)) & 1
    return b

# ================= END PATCH ===================

def random_qpsk_matrix(rows, cols, rng):
    """生成 rows×cols 的随机 QPSK（±1±j)/√2 矩阵"""
    re = rng.choice([-1, 1], size=(rows, cols))
    im = rng.choice([-1, 1], size=(rows, cols))
    return (re + 1j * im) / np.sqrt(2)


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


def calculate_papr(signal):
    """
    计算时域信号的PAPR
    :param signal: 时域信号数组（numpy array, 复数或实数）
    :return: PAPR（线性值）和 PAPR_dB（分贝）
    """
    power = np.abs(signal) ** 2
    peak_power = np.max(power)
    avg_power = np.mean(power)
    papr = peak_power / avg_power
    papr_db = 10 * np.log10(papr)
    print(f"Peak Power: {peak_power}, Average Power: {avg_power}")
    return papr, papr_db


def get_bits(file_pth: str):
    assert os.path.exists(file_pth), f"file not exist, given arg {file_pth}"
    with open(file_pth, 'rb') as f:
        byte_data = f.read()
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array)
    return bit_array.flatten()

def generate_chirp(fs, duration=2, f0=10, f1=24000):
    t = np.linspace(0, duration, int(fs * duration))
    return chirp(t, f0=f0, f1=f1, t1=duration, method='linear')


def OFDM_modulate_data(symbols, N, cp_len,front_guard_ratio=0.05, back_guard_ratio=0.2, fill_seed=RANDOM_FILL_SEED):
    """
    仅在正频(1..N/2-1)中使用前 use_ratio 部分承载数据，剩余用随机QPSK填充；DC与Nyquist=0；构造共轭对称。
    symbols: 1D QPSK 序列（来自映射后的数据）
    """
    # pos_cnt = N // 2 - 1  # 正频可用数（不含DC与Nyquist）
    # data_bins = int(np.floor(pos_cnt * use_ratio))
    # fill_bins = pos_cnt - data_bins  # 10% 填充
    # print(f"OFDM: N={N}, pos_cnt={pos_cnt}, data_bins={data_bins}, fill_bins={fill_bins}")
    # assert data_bins > 0 and fill_bins >= 0
    #
    # # 以 data_bins 为每符号承载量
    # num_symbols = len(symbols) // data_bins
    # if num_symbols == 0:
    #     return np.array([], dtype=float)
    # symbols = symbols[:num_symbols * data_bins]
    # data_matrix = symbols.reshape((num_symbols, data_bins))
    #
    # # 频域矩阵：先全 0
    # freq_data = np.zeros((num_symbols, N), dtype=complex)
    #
    # # 正频前 90%：承载数据（索引 1..data_bins）
    # freq_data[:, 1:1 + data_bins] = data_matrix
    #
    # # 正频后 10%：随机 QPSK 填充（索引 1+data_bins .. N/2-1）
    # if fill_bins > 0:
    #     rng = np.random.default_rng(fill_seed)
    #     freq_data[:, 1 + data_bins: 1 + pos_cnt] = random_qpsk_matrix(num_symbols, fill_bins, rng)
    #     # freq_data[:, 1 + data_bins: 1 + pos_cnt] = 0
    # # DC 与 Nyquist 置 0（已默认），构造负频为共轭对称
    # freq_data[:, N // 2 + 1:] = np.conj(freq_data[:, 1: N // 2])[:, ::-1]
    #
    # # IFFT + CP
    # time_data = np.fft.ifft(freq_data, axis=1)
    # cp = time_data[:, -cp_len:]
    # with_cp = np.hstack([cp, time_data]).flatten()
    # with_cp /= np.max(np.abs(with_cp) + 1e-12)
    # return np.real(with_cp)
    pos_cnt = N // 2 - 1  # 正频可用数（不含 DC 与 Nyquist）
    front_guard = int(np.floor(pos_cnt * front_guard_ratio))
    back_guard = int(np.floor(pos_cnt * back_guard_ratio))
    data_bins = pos_cnt - front_guard - back_guard
    print(f"pos_cnt={pos_cnt}, front_guard={front_guard}, back_guard={back_guard}, data_bins={data_bins}")
    if data_bins <= 0:
        raise ValueError(f"data_bins <= 0 (pos_cnt={pos_cnt}, front_guard={front_guard}, back_guard={back_guard}). "
                         "检查 N 或 guard_ratio 设置。")

    # 每个 OFDM 符号承载 data_bins 个有效子载波
    num_symbols = len(symbols) // data_bins
    remainder = len(symbols) % data_bins
    if remainder > 0:
        pad = data_bins - remainder
        symbols = np.concatenate([symbols, QPSK_mapping(np.random.randint(0, 2, size=pad * 2))])
        num_symbols += 1
    else:
        symbols = symbols[:num_symbols * data_bins]
    print(f"总数据符号数: {num_symbols} (每符号承载 {data_bins} 子载波)")

    if num_symbols == 0:
        return np.array([], dtype=float), np.zeros((0, N), dtype=complex)

    data_matrix = symbols.reshape((num_symbols, data_bins))

    # 构造 freq_data
    freq_data = np.ones((num_symbols, N), dtype=complex)

    # 数据区索引（正频）：[data_lo, data_hi)
    data_lo = 1 + front_guard
    data_hi = data_lo + data_bins  # 不包含

    # 放数据到中间 data_bins
    freq_data[:, data_lo:data_hi] = data_matrix

    # 前保护带（正频 1 .. data_lo-1）与后保护带（data_hi .. N/2-1）用随机 QPSK 填充
    rng = np.random.default_rng(fill_seed)
    left_guard_size = data_lo - 1  # 从 1 到 data_lo-1 一共 left_guard_size 个
    right_guard_size = (N // 2 - 1) - (data_hi - 1)  # 从 data_hi 到 N/2-1
    # #随机填充
    if left_guard_size > 0:
        freq_data[:, 1:1 + left_guard_size] = random_qpsk_matrix(num_symbols, left_guard_size, rng)
    if right_guard_size > 0:
        freq_data[:, data_hi:1 + pos_cnt] = random_qpsk_matrix(num_symbols, right_guard_size, rng)
    # if left_guard_size > 0:
    #     freq_data[:, 1:1 + left_guard_size] = 0
    # if right_guard_size > 0:
    #     freq_data[:, data_hi:1 + pos_cnt] = 0

    # 填充0或1
    # if left_guard_size > 0:
    #     freq_data[:, 1:1 + left_guard_size] = np.zeros((num_symbols, left_guard_size), dtype=complex)
    # if right_guard_size > 0:
    #     freq_data[:, data_hi:1 + pos_cnt] = np.zeros((num_symbols, right_guard_size), dtype=complex)
    # 负频按共轭对称
    freq_data[:, N // 2 + 1:] = np.conj(freq_data[:, 1:N // 2])[:, ::-1]
    time_data = np.fft.ifft(freq_data, axis=1)
    cp = time_data[:, -cp_len:]
    with_cp = np.hstack([cp, time_data]).flatten()

    # 归一化（避免除以零）
    max_abs = np.max(np.abs(with_cp))
    if max_abs > 0:
        with_cp = with_cp / max_abs

    return np.real(with_cp)

def OFDM_modulate_data_with_comb(
        symbols, N, cp_len, iteration=5, seed=128,
        front_guard_ratio=0.05, back_guard_ratio=0.2, fill_seed=2025
):
    """
    OFDM 调制 + 梳状导频插入（开头不插 pilot）。
    使用说明：
      - 在正频 1..N/2-1 中，保留前 front_guard_ratio 与后 back_guard_ratio
        作为保护带（用随机 QPSK 填充），中间区承载有效数据。
      - 其余逻辑（pilot 插入、IFFT、CP、归一化）不变。

    参数:
      symbols: 1D QPSK 数据符号（complex）
      N: IFFT 点数
      cp_len: CP 长度
      iteration: 每隔多少数据符号插入一个 pilot
      seed: pilot 基础种子
      front_guard_ratio: 正频前侧保护带比例（例如 0.05 表示前 5%）
      back_guard_ratio: 正频后侧保护带比例
      fill_seed: 随机填充的 RNG 种子
    返回:
      with_cp_real: 时域实数波形（1D）
      freq_with_pilot: 插入 pilot 后的 2D 频域矩阵（rows = 符号数, cols = N）
    """
    print(f"梳状导频模式（front_guard={front_guard_ratio * 100:.1f}%, back_guard={back_guard_ratio * 100:.1f}%），N = {N}")

    pos_cnt = N // 2 - 1  # 正频可用数（不含 DC 与 Nyquist）
    front_guard = int(np.floor(pos_cnt * front_guard_ratio))
    back_guard = int(np.floor(pos_cnt * back_guard_ratio))
    data_bins = pos_cnt - front_guard - back_guard
    print(f"pos_cnt={pos_cnt}, front_guard={front_guard}, back_guard={back_guard}, data_bins={data_bins}")
    if data_bins <= 0:
        raise ValueError(f"data_bins <= 0 (pos_cnt={pos_cnt}, front_guard={front_guard}, back_guard={back_guard}). "
                         "检查 N 或 guard_ratio 设置。")

    # 每个 OFDM 符号承载 data_bins 个有效子载波
    num_symbols = len(symbols) // data_bins
    remainder = len(symbols) % data_bins
    if remainder > 0:
        pad = data_bins - remainder
        symbols = np.concatenate([symbols, QPSK_mapping(np.random.randint(0, 2, size=pad * 2))])
        num_symbols += 1
    else:
        symbols = symbols[:num_symbols * data_bins]
    print(f"总数据符号数: {num_symbols} (每符号承载 {data_bins} 子载波)")

    if num_symbols == 0:
        return np.array([], dtype=float), np.zeros((0, N), dtype=complex)

    data_matrix = symbols.reshape((num_symbols, data_bins))

    # 构造 freq_data
    freq_data = np.ones((num_symbols, N), dtype=complex)

    # 数据区索引（正频）：[data_lo, data_hi)
    data_lo = 1 + front_guard
    data_hi = data_lo + data_bins  # 不包含

    # 放数据到中间 data_bins
    freq_data[:, data_lo:data_hi] = data_matrix

    # 前保护带（正频 1 .. data_lo-1）与后保护带（data_hi .. N/2-1）用随机 QPSK 填充
    rng = np.random.default_rng(fill_seed)
    left_guard_size = data_lo - 1  # 从 1 到 data_lo-1 一共 left_guard_size 个
    right_guard_size = (N // 2 - 1) - (data_hi - 1)  # 从 data_hi 到 N/2-1
    # #随机填充
    if left_guard_size > 0:
        freq_data[:, 1:1 + left_guard_size] = random_qpsk_matrix(num_symbols, left_guard_size, rng)
    if right_guard_size > 0:
        freq_data[:, data_hi:1 + pos_cnt] = random_qpsk_matrix(num_symbols, right_guard_size, rng)
    # if left_guard_size > 0:
    #     freq_data[:, 1:1 + left_guard_size] = 0
    # if right_guard_size > 0:
    #     freq_data[:, data_hi:1 + pos_cnt] = 0

    # 填充0或1
    # if left_guard_size > 0:
    #     freq_data[:, 1:1 + left_guard_size] = np.zeros((num_symbols, left_guard_size), dtype=complex)
    # if right_guard_size > 0:
    #     freq_data[:, data_hi:1 + pos_cnt] = np.zeros((num_symbols, right_guard_size), dtype=complex)
    # 负频按共轭对称
    freq_data[:, N // 2 + 1:] = np.conj(freq_data[:, 1:N // 2])[:, ::-1]

    # ===== 插入梳状导频（开头不插 pilot），插入规则与原逻辑相同 =====
    freq_with_pilot_list = []
    pilot_counter = 0
    i = 0
    while i < num_symbols:
        # 插入 iteration 个数据符号（或剩余的全部）
        for j in range(iteration):
            if i >= num_symbols:
                break
            freq_with_pilot_list.append(freq_data[i])
            i += 1
        # 插入一个 pilot（如果还没到最后）
        if i < num_symbols:
            freq_with_pilot_list.append(generate_pilot_symbol(N, seed + pilot_counter))
            pilot_counter += 1

    freq_with_pilot = np.vstack(freq_with_pilot_list)  # shape = (num_symbols + num_pilots, N)
    print(f"插入导频后总符号数: {freq_with_pilot.shape[0]} (原数据符号数: {num_symbols})")

    # ===== IFFT + CP =====
    time_data = np.fft.ifft(freq_with_pilot, axis=1)
    cp = time_data[:, -cp_len:]
    with_cp = np.hstack([cp, time_data]).flatten()

    # 归一化（避免除以零）
    max_abs = np.max(np.abs(with_cp))
    if max_abs > 0:
        with_cp = with_cp / max_abs

    return np.real(with_cp), freq_with_pilot


# ---------------------- OFDM modulation functions ----------------------


def QPSK_mapping(bits):
    bits = bits.reshape((-1, 2))
    mapping_table = {(0, 0): 1 + 1j, (0, 1): -1 + 1j, (1, 0): 1 - 1j, (1, 1): -1 - 1j}
    syms = np.array([mapping_table[tuple(b)] for b in bits])
    return syms / np.sqrt(2)


def generate_pilot_symbol(N, seed=256):
    """
    生成一个对称的随机导频符号序列
    N: 子载波数
    seed: 随机种子，保证相同seed时生成的导频相同
    """
    rng = np.random.default_rng(seed)  # 新的随机数生成器，不影响全局
    half = N // 2

    real_parts = rng.choice([-1, 1], size=half - 1)
    imag_parts = rng.choice([-1, 1], size=half - 1)
    X_half = (real_parts + 1j * imag_parts) / np.sqrt(2)

    # qpsk = rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], size=half - 1)

    X_freq = np.zeros(N, dtype=complex)
    X_freq[0] = 1
    X_freq[1:half] = X_half
    # X_freq[1:half] = qpsk
    X_freq[half] = 1
    X_freq[half + 1:] = np.conj(X_half[::-1])
    # X_freq[half + 1:] = np.conj(qpsk[::-1])
    return X_freq



def generate_pilot_combed_symbol(N, seed=256, iterations=10, num_of_data_symbols=50, block_size=4):
    """
    生成梳状导频（一维拼接）
    N: 子载波数
    seed: 初始随机种子
    iterations: 每隔多少个数据符号插入一个导频
    num_of_data_symbols: 总共有多少个数据符号
    block_size: 每个导频块的大小（即每个导频符号的长度）
    """
    num_pilots = int(np.ceil(num_of_data_symbols / iterations)) * block_size

    # 直接拼成一维向量
    pilot_seq = np.concatenate([generate_pilot_symbol(N, seed + i) for i in range(num_pilots)])
    return pilot_seq


def ofdm_modulate(symbol_freq):
    time_signal = np.fft.ifft(symbol_freq)
    return np.concatenate([time_signal[-cp_len:], time_signal])


# ---------------------- Main flow ----------------------
def main():
    # Pilots
    chirp_sig = generate_chirp(fs, f0=10, f1=24000)
    chirp_tail = generate_chirp(fs, f0=20, f1=18000)
    chirp_cof = 0.4

    tx_signal = np.array([])

    mode = input("mode:1different,2same")
    if mode == '1':
        pilot_different = []
        for i in range(num_symbols):
            pilot = generate_pilot_symbol(N_fft, seed=256 + i)
            pilot_different.append(pilot)
            ofdm_time = ofdm_modulate(pilot)
            tx_signal = np.concatenate([tx_signal, ofdm_time])
        np.save(PILOT_SAVE_PATH, pilot_different)
    else:
        pilot = generate_pilot_symbol(N_fft, seed=256)
        ofdm_time = ofdm_modulate(pilot)
        pilot_same = []
        for _ in range(num_symbols):
            pilot_same.append(pilot)
            tx_signal = np.concatenate([tx_signal, ofdm_time])
        np.save(PILOT_SAVE_PATH, pilot_same)

    tx_signal_real = np.real(tx_signal)
    tx_signal_real /= np.max(np.abs(tx_signal_real))
    ch = input("1two chirp 2one chirp")
    if ch == '1':
        tx_signal_real = np.concatenate([chirp_sig*chirp_cof, tx_signal_real, chirp_tail*chirp_cof])
    else:
        tx_signal_real = np.concatenate([chirp_sig*chirp_cof, tx_signal_real])
        # tx_signal_real = np.concatenate([chirp_sig , tx_signal_real])

    file_path = TIFF_INPUT_PATH  # 也可以切换为 TXT_INPUT_PATH / PNG_INPUT_PATH

    ext = Path(file_path).suffix.lower()
    if ext in {'.txt'}:
        bits = get_bits(file_path)
    elif ext in {'.tif', '.tiff', '.bin', ''}:
        # 二进制按字节读→bit 即可（复用 get_bits_from_txt 的实现没问题）
        bits = get_bits(file_path)
    else:
        # 默认按二进制读取
        bits = get_bits(file_path)

    print(f"raw bits: {bits.shape}")

    # ========= 24-bit 类型 + 40-bit 大小 的 64bit 头 =========
    # 仅支持 3 字符类型。常见映射如下（不在表内则用 'bin'）：
    type_map = {
        '.txt': 'txt',
        '.tif': 'tif',
        '.tiff': 'tif',
        '.png': 'png',
    }
    file_type = type_map.get(ext, 'bin')  # 必须是 3 字符
    hdr_type = ascii3_to_24bits(file_type)
    print(f"file head type: {hdr_type}")

    payload_bit_len = int(len(bits))
    hdr_size = u40_to_bits_msb(payload_bit_len)  # 40 bit
    print(f"payload bit size: {hdr_size}")

    header_64bits = np.concatenate([hdr_type, hdr_size])  # 24 + 40 = 64
    bits = np.concatenate([header_64bits, bits])
    print(f"[TX] header file_type='{file_type}', payload_len(bits)={payload_bit_len}")
    print(f"total bits (with 64-bit header): {len(bits)}")
    # bits_scr = scrambler(bits).astype(np.uint8)
    # bits_scr = scrambler_random(bits, seed=256)
    bits_scr = bits.copy()
    # print(f"scrambled bits (first 16): {bits_scr[:16]}")

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
    ofdm_mode = input("ofdm_modulate_mode:1.comb pilot 2.no comb pilot")
    if ofdm_mode == '1':
        data_waveform, freq_data = OFDM_modulate_data_with_comb(qpsk_symbols, N_fft, cp_len, iteration=10, seed=128)
        print(f"length of freq_data with comb pilot: {freq_data.shape}")
    else:
        data_waveform = OFDM_modulate_data(qpsk_symbols, N_fft, cp_len)

    # 计算有效数据的 PAPR
    # papr_linear, papr_db = calculate_papr(data_waveform)
    # print(f"PAPR (线性): {papr_linear:.3f}, PAPR (dB): {papr_db:.2f} dB")
    # 合成最终发送波形
    tx_signal_realtime = np.concatenate([tx_signal_real, data_waveform])

    # 归一化后播放
    signal_cut = tx_signal_realtime.copy()
    signal_cut /= np.max(np.abs(signal_cut))
    signal_cut = np.concatenate([signal_cut, chirp_tail*chirp_cof])
    # signal_cut = prepend_silence_complex(signal_cut, fs=48000, silence_sec=10)
    tx_signal_realtime = signal_cut.copy()
    print(f"Final transmit signal length: {len(tx_signal_realtime)} samples")
    print(f"Duration: {len(tx_signal_realtime) / fs:.2f} seconds")
    # plt.figure()
    # plt.plot(tx_signal_realtime)
    # plt.title("Transmit Signal with LDPC-coded payload")
    # plt.xlabel("Sample Index")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.show()

    print("🔊 Playing the transmit signal...")
    # sd.play(signal_cut, fs)
    # sd.wait()
    np.save(DATA_WAVEFORM_SAVE_PATH, signal_cut)

    # 对比图（若未来加入削峰/滤波，可在此对比）
    plt.figure(figsize=(12, 6))
    # plot_start = 1000000
    plt.plot(tx_signal_realtime, label='TX (LDPC-coded)')
    plt.legend()
    plt.grid(True)
    plt.show()

    if WAV_SAVE_PATH:
        write(WAV_SAVE_PATH, fs, (signal_cut * 32767).astype(np.int16))

    print("✅ Transmission (LDPC-coded) completed")


if __name__ == "__main__":
    main()