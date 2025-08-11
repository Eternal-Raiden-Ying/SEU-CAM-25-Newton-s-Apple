# 录音播放部分
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import chirp, butter, filtfilt, lfilter
import matplotlib.pyplot as plt
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import scrambler
# 读取txt文件并转换为比特流
def get_bits_from_txt(file_pth: str):
    """
        read file in binary and return a binary np.ndarray (flattened)
    :param file_pth: just file path
    :return: binary np.ndarray, like [0,0,0,1,1,1,.....]
    """
    assert os.path.exists(file_pth), f"file not exist, given arg {file_pth}"

    with open(file_pth, 'rb') as file:
        byte_data = file.read()

    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array)
    return bit_array.flatten()

# Scrambler function to randomize bits


# Generate a linear chirp signal
def generate_chirp(fs, duration=2, f0=10, f1=24000):
    t = np.linspace(0, duration, int(fs * duration))
    chirp_sig = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    return chirp_sig

# Symbol数据调制到时域
def OFDM_modulate_data(symbols, N, cp_len):
    num_symbols = len(symbols) // (N // 2 - 1)
    symbols = symbols[:num_symbols * (N // 2 - 1)]
    data_matrix = symbols.reshape((num_symbols, N // 2 - 1))

    # Initialize freq_data as complex to avoid casting warnings
    freq_data = np.ones((num_symbols, N), dtype=complex)
    freq_data[:, 1:N // 2] = data_matrix
    freq_data[:, N // 2 + 1:] = np.conj(data_matrix)[:, ::-1]  # Hermitian symmetry
    print(f"freq_data shape: {freq_data.shape}")
    print(f"freq_data first 10: {freq_data[:10]}")
    print(f"freq_data last 10: {freq_data[-10:]}")
    time_data = np.fft.ifft(freq_data, axis=1)
    cp = time_data[:, -cp_len:]
    with_cp = np.hstack([cp, time_data])
    with_cp_flatten = with_cp.flatten()
    print(f"with_cp_flatten : {with_cp_flatten}")
    with_cp_flatten /= np.max(np.abs(with_cp_flatten))  # Normalize
    return np.real(with_cp_flatten)

def QPSK_mapping(bits):
    bits = bits.reshape((-1, 2))
    mapping_table = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 0): 1 - 1j,
        (1, 1): -1 - 1j
    }
    symbols = np.array([mapping_table[tuple(b)] for b in bits])
    return symbols / np.sqrt(2)  # 归一化

# Parameters
fs = 48000
N = 4096
cp_len = 1024
num_symbols = 8

# Generate pilot symbol (e.g., using BPSK or QPSK)
def generate_pilot_symbol(N):
    half = N // 2
    real_parts = np.random.choice([-1, 1], size=half - 1)
    imag_parts = np.random.choice([-1, 1], size=half - 1)
    X_half = (real_parts + 1j * imag_parts) / np.sqrt(2)  # Normalize to unit energy

    X_freq = np.zeros(N, dtype=complex)
    X_freq[0] = 1  # DC component
    X_freq[1:half] = X_half
    X_freq[half] = 1  # Nyquist frequency (real)
    X_freq[half + 1:] = np.conj(X_half[::-1])  # Hermitian symmetry
    return X_freq

# Perform OFDM modulation with IFFT and cyclic prefix
def ofdm_modulate(symbol_freq):
    time_signal = np.fft.ifft(symbol_freq)
    return np.concatenate([time_signal[-cp_len:], time_signal])

# Generate RRC filter
# def rrc(beta, span, sps):
#     # beta: roll-off, span in symbols, sps samples per symbol
#     N = span * sps
#     t = np.arange(-N/2, N/2 + 1) / float(sps)
#     h = np.zeros_like(t)
#     for i, ti in enumerate(t):
#         if ti == 0.0:
#             h[i] = 1.0 - beta + 4*beta/np.pi
#         elif abs(abs(ti) - 1/(4*beta)) < 1e-8:
#             h[i] = (beta/np.sqrt(2)) * ((1+2/np.pi) *
#                     (np.sin(np.pi/(4*beta))) + (1-2/np.pi) * (np.cos(np.pi/(4*beta))))
#         else:
#             numerator = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
#             denominator = np.pi*ti*(1-(4*beta*ti)**2)
#             h[i] = numerator / denominator
#     return h / np.sqrt(np.sum(h**2))  # normalize

# Generate the chirp signal for prefix
chirp_sig = generate_chirp(fs, f0=10, f1=24000)
chirp_tail = generate_chirp(fs, f0=20, f1=24000)

tx_signal = np.array([])
i = input("mode:1different,2same")  # input得到的类型为str
output_path = os.path.join(r"D:\Pycharm\PythonProject1\save\pilot_different_txt11.npy")

if i == '1':
    pilot_different = []
    for _ in range(num_symbols):
        pilot = generate_pilot_symbol(N)
        pilot_different.append(pilot)  # Collect different pilots
        ofdm_time = ofdm_modulate(pilot)
        tx_signal = np.concatenate([tx_signal, ofdm_time])
    np.save(output_path, pilot_different)  # Save the list of pilots
else:
    pilot = generate_pilot_symbol(N)
    ofdm_time = ofdm_modulate(pilot)
    np.save(output_path, pilot)
    print(f"pilot shape: {pilot.shape}")
    for _ in range(num_symbols):
        tx_signal = np.concatenate([tx_signal, ofdm_time])

# Convert to real signal and normalize
tx_signal_real = np.real(tx_signal)
tx_signal_real /= np.max(np.abs(tx_signal_real))  # Normalize
i = input("1two chirp 2one chirp")
if i == '1':
    tx_signal_real = np.concatenate([chirp_sig, tx_signal_real])
    tx_signal_real = np.concatenate([tx_signal_real, chirp_tail])
else:
    tx_signal_real = np.concatenate([chirp_sig, tx_signal_real])

# def clipping_filtering(signal, fs, clipping_threshold_ratio=0.4, filter_order=5, cutoff_freq=20000):
#     """
#     Clipping and Filtering for PAPR reduction
#     :param signal: input time domain signal (1D numpy array)
#     :param fs: sampling frequency
#     :param clipping_threshold_ratio: clipping threshold relative to max amplitude (0~1)
#     :param filter_order: order of Butterworth filter
#     :param cutoff_freq: cutoff frequency of low-pass filter (Hz)
#     :return: clipped and filtered signal
#     """
#     max_amp = np.max(np.abs(signal))
#     threshold = clipping_threshold_ratio * max_amp
#     clipped_signal = np.clip(signal, -threshold, threshold)
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff_freq / nyq
#     b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
#     filtered_signal = filtfilt(b, a, clipped_signal)
#     return filtered_signal

# 使用示例
# Read bits from a text file
bits = get_bits_from_txt(r"D:\55495\个人文件\剑桥\OFDM\shakespace(short).txt")
print(f"bits shape: {bits.shape}")
print(f"bits first 10: {bits[:10]}")  # Print first 10 bits for debugging

# Apply scrambler to randomize bits
scrambled_bits = scrambler(bits)
print(f"scrambled_bits first 10: {scrambled_bits[:10]}")

# QPSK mapping
qpsk_symbols = QPSK_mapping(scrambled_bits)
print(f"symbols shape: {qpsk_symbols.shape}")
print(f"symbols first 10: {qpsk_symbols[:10]}")
data_waveform = OFDM_modulate_data(qpsk_symbols, N, cp_len)
print(f"data_Waveform_length: {len(data_waveform)}")
txt_time = np.array(data_waveform)
np.save(r"D:\Pycharm\PythonProject1\save\1"
        r"txt_time.npy", txt_time)

# 削峰并滤波
clipped_filtered_signal = data_waveform
clipped_filtered_signal /= np.max(np.abs(clipped_filtered_signal))  # Normalize

# 加入RRC滤波部分
# beta = 0.25
# span = 8
# sps = 1  # 由于已经是时域采样信号，使用sps=1应用滤波
# h = rrc(beta, span, sps)
# clipped_filtered_signal = lfilter(h, 1.0, clipped_filtered_signal)
# clipped_filtered_signal /= np.max(np.abs(clipped_filtered_signal))  # 重新归一化

# Concatenate the chirp and transmit signal
tx_signal_realtime = np.concatenate([tx_signal_real, data_waveform])
signal_cut = np.concatenate([tx_signal_real, clipped_filtered_signal])

# Plot the signal
plt.plot(tx_signal_realtime)
plt.title("Transmit Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Play the signal
print("🔊 Playing the transmit signal...")
sd.play(signal_cut, fs)
sd.wait()

# 画图对比
plt.figure(figsize=(12, 6))
plt.plot(tx_signal_realtime, label='Original Signal')
plt.plot(signal_cut, label='Clipped & Filtered Signal', alpha=1.0)
plt.title('Clipping and Filtering to Reduce PAPR')
plt.legend()
plt.grid(True)
plt.show()

# Save a WAV file
# write(r"D:\Pycharm\PythonProject1\record\tx_signal.wav", fs, (tx_signal_real * 32767).astype(np.int16))
print("✅ Transmission completed")