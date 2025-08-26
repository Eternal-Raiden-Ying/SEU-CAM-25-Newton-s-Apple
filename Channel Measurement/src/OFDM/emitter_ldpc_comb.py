import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import chirp, butter, filtfilt, lfilter
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import LDPC_PY_PATH
from utils import ldpc_encode_bits,scrambler, scrambler_random

# ---------- LDPC library import and params (match TX) ----------
if LDPC_PY_PATH and LDPC_PY_PATH not in sys.path:
    sys.path.append(LDPC_PY_PATH)

try:
    import ldpc  # from ldpc_jossy/py
except Exception as e:
    raise ImportError(
        f"无法导入 ldpc 包：{e}\n"
        f"请检查 LDPC_PY_PATH 是否指向 ldpc_jossy/py，并确保已按 README 编译了解码动态库。"
    )

# ---------------------- Parameters ----------------------
fs = 48000
N_fft = 8192
cp_len = 1024
num_symbols = 8

# LDPC parameters (defaults are safe; change to your needs)
LDPC_STANDARD = '802.11n'  # '802.11n' or '802.16'
LDPC_RATE = '1/2'  # '1/2','2/3','3/4','5/6'
LDPC_Z = 27  # for 802.11n usually 27/54/81
LDPC_PTYPE = 'A'  # only used for 802.16 rate 2/3 or 3/4

# ---------------------- I/O paths (adjust if needed) ----------------------
TXT_INPUT_PATH = r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\data\shakespace(short).txt"
PNG_INPUT_PATH = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\data\cambridge.png"
PILOT_SAVE_PATH = r"pilot.npy"


# DATA_WAVEFORM_SAVE_PATH = r"C:\Users\Lenovo\Desktop\Cambridge_Summer_project\signal_different_txt820_1.npy"
# WAV_SAVE_PATH =  r"C:\Users\Lenovo\Desktop\Cambridge_Summer_project\tx_signal_1.wav"

# ---------------------- Helpers ----------------------

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
    return papr, papr_db

def get_bits_from_png(file_pth: str):
    assert os.path.exists(file_pth), f"file not exist, given arg {file_pth}"
    with open(file_pth, 'rb') as f:
        byte_data = f.read()
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array)
    return bit_array.flatten()

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
    print(f"num of symbols:{num_symbols}")
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

def OFDM_modulate_data_with_comb(symbols, N, cp_len, iteration=10, seed=128):
    """
    OFDM 调制 + 梳状导频插入（开头不插 pilot）
    :param symbols: 原始数据符号（一维数组）
    :param N: IFFT 点数
    :param cp_len: 循环前缀长度
    :param iteration: 每隔多少个数据符号插入一个 pilot
    :param seed: 初始随机种子
    :return: 1D 实数时域信号, 插入 pilot 后的 freq_data 矩阵
    """
    # 每个 OFDM 符号可承载的数据子载波数
    K = N // 2 - 1
    num_symbols = len(symbols) // K
    symbols = symbols[:num_symbols * K]
    data_matrix = symbols.reshape((num_symbols, K))

    # 构造原始 freq_data 矩阵
    freq_data = np.ones((num_symbols, N), dtype=complex)
    freq_data[:, 1:N // 2] = data_matrix
    freq_data[:, N // 2 + 1:] = np.conj(data_matrix)[:, ::-1]  # Hermitian symmetry

    # ===== 插入梳状导频（开头不插 pilot） =====
    freq_with_pilot = []
    pilot_counter = 0
    i = 0
    while i < num_symbols:
        # 先插入 iteration 个数据符号（或剩余的全部）
        for j in range(iteration):
            if i >= num_symbols:
                break
            freq_with_pilot.append(freq_data[i])
            i += 1
        # 插入一个 pilot（如果还没到最后）
        if i < num_symbols:
            freq_with_pilot.append(generate_pilot_symbol(N, seed + pilot_counter))
            pilot_counter += 1

    # 转换成 numpy 矩阵
    freq_with_pilot = np.vstack(freq_with_pilot)  # shape = (num_pilots + num_symbols, N)

    # ===== IFFT + CP =====
    time_data = np.fft.ifft(freq_with_pilot, axis=1)
    cp = time_data[:, -cp_len:]
    with_cp = np.hstack([cp, time_data]).flatten()
    # 归一化
    with_cp /= np.max(np.abs(with_cp))
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

    X_freq = np.zeros(N, dtype=complex)
    X_freq[0] = 1
    X_freq[1:half] = X_half
    X_freq[half] = 1
    X_freq[half + 1:] = np.conj(X_half[::-1])
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
        np.save(PILOT_SAVE_PATH, pilot)
        for _ in range(num_symbols):
            tx_signal = np.concatenate([tx_signal, ofdm_time])

    tx_signal_real = np.real(tx_signal)
    tx_signal_real /= np.max(np.abs(tx_signal_real))
    ch = input("1two chirp 2one chirp")
    if ch == '1':
        tx_signal_real = np.concatenate([chirp_sig, tx_signal_real, chirp_tail])
    else:
        tx_signal_real = np.concatenate([chirp_sig, tx_signal_real])

    # 读取 & 扰码
    # bits = get_bits_from_png(PNG_INPUT_PATH)
    bits = get_bits_from_txt(TXT_INPUT_PATH)
    print(f"raw bits: {bits.shape}")
    # bits_scr = scrambler(bits).astype(np.uint8)
    bits_scr = scrambler_random(bits, seed=256)
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
    ofdm_mode = input("ofdm_modulate_mode:1.comb pilot 2.no comb pilot")
    if ofdm_mode == '1':
        data_waveform, freq_data = OFDM_modulate_data_with_comb(qpsk_symbols, N_fft, cp_len, iteration=5, seed=128)
    else:
        data_waveform = OFDM_modulate_data(qpsk_symbols, N_fft, cp_len)

    # 计算有效数据的 PAPR
    papr_linear, papr_db = calculate_papr(data_waveform)
    print(f"PAPR (线性): {papr_linear:.3f}, PAPR (dB): {papr_db:.2f} dB")
    # 合成最终发送波形
    tx_signal_realtime = np.concatenate([tx_signal_real, data_waveform])

    # 归一化后播放
    signal_cut = tx_signal_realtime.copy()
    signal_cut /= np.max(np.abs(signal_cut))
    signal_cut = np.concatenate([signal_cut, chirp_tail])
    tx_signal_realtime = signal_cut

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
    np.save(r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\save\signalwhole_signal_821.npy",signal_cut)

    # 对比图（若未来加入削峰/滤波，可在此对比）
    plt.figure(figsize=(12, 6))
    plt.plot(tx_signal_realtime, label='TX (LDPC-coded)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # if WAV_SAVE_PATH:
    #     write(WAV_SAVE_PATH, fs, (signal_cut * 32767).astype(np.int16))

    print("✅ Transmission (LDPC-coded) completed")

if __name__ == "__main__":
    main()