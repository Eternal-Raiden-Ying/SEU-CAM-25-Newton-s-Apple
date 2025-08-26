import numpy as np
from scipy import signal
import sounddevice as sd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 防止负号乱码
from scipy.signal import chirp, correlate, find_peaks


from utils import QPSK_mapping
#%%
# 系统参数
fs = 48000
N = 4096  # 总子载波数
K = 24   # 导频间隔（每24个子载波一个导频）
pilot_indices = np.arange(1, N//2, K)  # 导频子载波索引: 0, 6, 12, ...
data_indices = np.setdiff1d(np.arange(1,N//2), pilot_indices)  # 数据子载波索引
cp_length = 128  # 循环前缀长度
qpsk_symbols = [1+1j, 1-1j, -1+1j, -1-1j] / np.sqrt(2)  # 归一化QPSK符号

def generate_chirp(fs, duration=1, f0=10, f1=24000):
    t = np.linspace(0, duration, int(fs * duration))
    chirp_sig = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    return chirp_sig

chirp_sig = generate_chirp(fs)

# Hamming码编码
def hamming_encode(data_bits):
    # 输入：32位数据，输出：38位Hamming码
    m = 6  # 校验位数，2^6=64 > 32+6+1
    n = 32 + m  # 总长度
    codeword = np.zeros(n, dtype=int)
    # 放置数据位（非2的幂位置：3,5,6,7,9,...）
    data_idx = [i for i in range(n) if i not in [0, 1, 2, 4, 8, 16, 32]]
    for i, idx in enumerate(data_idx):
        codeword[idx] = data_bits[i]
    # 计算校验位
    for i in range(m):
        check_pos = 2**i
        parity = 0
        for j in range(n):
            if (j + 1) & (check_pos):  # 检查位覆盖的索引
                parity ^= codeword[j]
        codeword[check_pos - 1] = parity
    return codeword

# 读取txt文件并编码
def read_and_encode_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # 文本比特（ASCII）
    text_bits = ''.join(format(ord(char), '08b') for char in text)
    text_bits = np.array([int(b) for b in text_bits])

    # 文件头：文件名（最大32字节）+ 分隔符 + 文件大小（32位）+ 分隔符
    #TODO: 文件头的字长固定， 感觉没有必要填充
    filename = file_path.split('\\')[-1].encode('utf-8')[:32].ljust(32, b'\x00')
    filesize = len(text)  # 字节数
    filesize_bits = np.array([int(b) for b in f"{filesize:032b}"])

    # Hamming编码
    filename_bits = np.unpackbits(np.frombuffer(filename, dtype=np.uint8))
    filename_hamming = []
    for i in range(0, 256, 32):  # 分8块，每32位编码
        block = filename_bits[i:i+32]
        filename_hamming.append(hamming_encode(block))
    filename_hamming = np.concatenate(filename_hamming)  # 256→304位
    filesize_hamming = hamming_encode(filesize_bits)  # 32→38位

    # 文件头：文件名(304位) + 分隔符(8位) + 大小(38位) + 分隔符(8位)
    separator = np.zeros(8, dtype=int)  # b'\x00'
    header_bits = np.concatenate([filename_hamming, separator, filesize_hamming, separator])

    # 总比特流：文件头 + 文本
    bits = np.concatenate([header_bits, text_bits])
    return bits

# 生成Zadoff-Chu导频序列
def generate_zadoff_chu(length, u=1):
    n = np.arange(length)
    return np.exp(-1j * np.pi * u * n * (n + 1) / length)

# QPSK调制
def qpsk_modulate(bits):
    symbols = []
    for i in range(0, len(bits), 2):
        bit_pair = bits[i:i+2]
        if len(bit_pair) == 2:
            idx = int(bit_pair[0]) * 2 + int(bit_pair[1])
            symbols.append(qpsk_symbols[idx])
    return np.array(symbols)

# 插入导频和数据
def insert_pilot_and_data(data_symbols, pilot_symbols):
    ofdm_symbol = np.zeros(N, dtype=complex)
    # 插入导频
    ofdm_symbol[pilot_indices] = pilot_symbols
    # 插入数据
    ofdm_symbol[data_indices] = data_symbols
    ofdm_symbol[N//2] = 1
    ofdm_symbol[0] = 1
    ofdm_symbol[N//2 + 1:] = np.conjugate(ofdm_symbol[1:N//2])[::-1]
    return ofdm_symbol

# 添加循环前缀
def add_cyclic_prefix(ofdm_time):
    return np.concatenate([ofdm_time[-cp_length:], ofdm_time])

# 主函数
def transmitter(file_path):
    bits = read_and_encode_text(file_path)
    data_symbols = QPSK_mapping(bits.reshape(-1,2))
    pilot_symbols = generate_zadoff_chu(len(pilot_indices))
    ofdm_symbols = []
    num_data_per_symbol = len(data_indices)
    for i in range(0, len(data_symbols), num_data_per_symbol):
        current_data = data_symbols[i:i+num_data_per_symbol]
        if len(current_data) < num_data_per_symbol:
            current_data = np.concatenate([current_data, np.ones(num_data_per_symbol-current_data.size)])
        ofdm_freq = insert_pilot_and_data(current_data, pilot_symbols)
        ofdm_time = np.fft.ifft(ofdm_freq, N)
        ofdm_with_cp = add_cyclic_prefix(ofdm_time)
        ofdm_symbols.append(ofdm_with_cp)
    tx_signal = np.concatenate(ofdm_symbols)
    tx_signal_real = np.real(tx_signal)
    tx_signal_real /= np.max(np.abs(tx_signal_real))
    chirp_sig = generate_chirp(fs)
    tx_signal_real = np.append(chirp_sig, tx_signal_real)
    return tx_signal_real * 0.9
#%%
# 示例调用
if __name__ == "__main__":

    file_path = r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\data\shakespace_poem.txt"  # 输入txt文件
    tx_signal = transmitter(file_path)
    print("Generated TX signal length:", len(tx_signal))
    # 保存信号（可选）
    print("🔊 Playing the transmit signal...")
    sd.play(tx_signal, fs)
    sd.wait()
    print("✅ Transmission completed")
    np.save(r"D:\Pycharm\PythonProject1\record\tx_signal.npy", tx_signal)

    # Plot the signal
    plt.plot(tx_signal)
    plt.title("Transmit Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
