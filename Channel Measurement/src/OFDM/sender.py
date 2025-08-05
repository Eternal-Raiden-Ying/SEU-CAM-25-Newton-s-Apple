import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import chirp

def generate_chirp(fs, duration=1, f0=10, f1=24000):
    t = np.linspace(0, duration, int(fs * duration))
    chirp_sig = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    return chirp_sig

# 参数设置
fs = 48000  # 采样率
N = 1024    # OFDM 子载波数
cp_len = 256  # 循环前缀长度
num_symbols = 10  # OFDM 符号数

chirp_sig = generate_chirp(fs)

def generate_pilot_symbol(N):
    # 只生成前半部分（不含DC和Nyquist）
    half = N // 2
    real_parts = np.random.choice([-1, 1], size=half-1)
    imag_parts = np.random.choice([-1, 1], size=half-1)

# 构造复数序列
    X_half = real_parts + 1j * imag_parts

    # 构造频域：DC + X_half + Nyquist + 共轭对称部分
    X_freq = np.zeros(N, dtype=complex)
    X_freq[0] = 1  # DC 分量（实数）
    X_freq[1:half] = X_half
    X_freq[half] = 1  # Nyquist 分量（实数）
    X_freq[half+1:] = np.conj(X_half[::-1])
    return X_freq

# IFFT生成OFDM符号
def ofdm_modulate(symbol_freq):
    time_signal = np.fft.ifft(symbol_freq)
    # 添加循环前缀
    return np.concatenate([time_signal[-cp_len:], time_signal])

X_freq = generate_pilot_symbol(1024)
print(f"频域导频符号：{X_freq}, 长度={len(X_freq)}")
x_time = np.fft.ifft(X_freq)
print(f"时域信号：{x_time}, 长度={len(x_time)}")

print("时域信号最大虚部大小：", np.max(np.abs(np.imag(x_time))))

# 合成完整信号
tx_signal = np.array([])
for _ in range(num_symbols):
    pilot = generate_pilot_symbol(N)
    print(f"生成的导频符号：{pilot}, 长度={len(pilot)}")
    ofdm_time = ofdm_modulate(pilot)
    print(f"OFDM 符号时域信号：{ofdm_time}, 长度={len(ofdm_time)}")
    tx_signal = np.concatenate([tx_signal, ofdm_time])

# 正规化为实数并播放
print(f"tx_signal_real:{tx_signal}, 长度={len(tx_signal)}")
tx_signal_real = np.real(tx_signal)

tx_signal_real = np.concatenate([chirp_sig, tx_signal_real])
# tx_signal_real /= np.max(np.abs(tx_signal_real))  # 归一化

print("🔊 播放发送信号中...")
sd.play(tx_signal_real, fs)
sd.wait()

