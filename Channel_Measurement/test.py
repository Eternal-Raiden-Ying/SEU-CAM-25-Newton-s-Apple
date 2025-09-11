import os
import numpy as np

from module.utils.modulate import QPSK_mapping,OFDM_modulate

print(np.linspace(16,27,12).astype(int))

# def generate_8190bit_probe():
#     """ 生成8190比特CAZAC-QPSK信道探测序列 """
#     total_bits = 8190
#     n_symbols = total_bits // 2
#     r = 17  # 与4095互质的根索引
#
#     # 生成CAZAC序列
#     n = np.arange(n_symbols)
#     cazac = np.exp(-1j * np.pi * r * n * (n + 1) / n_symbols)
#
#     # QPSK量化
#     bits = np.zeros(total_bits, dtype=int)
#     for i in range(n_symbols):
#         bits[2 * i + 1] = 0 if np.real(cazac[i]) >= 0 else 1
#         bits[2 * i ] = 0 if np.imag(cazac[i]) >= 0 else 1
#     print ('df', len (bits))
#     return bits
#
#
#
# file_pth = r"D:\Documents\Coding\Python\SEUCAM\Channel_Measurement\record\noise_original.bin"
# with open(file_pth, 'rb') as file:
#     data = file.read()
#
#
# bits = generate_8190bit_probe()
#
# const = QPSK_mapping(bits.reshape(-1,2))
# pilot = OFDM_modulate(const, N=8192, cp_len=1024)
# print(pilot)


# # stable_record_to_npy.py
#
# import numpy as np
# import sounddevice as sd
#
# fs = 48000
# seconds = 50
# device = 9   # ← 用你清单里的“输入设备”索引；推荐 WASAPI 的 12
#
# # 可选：尝试 WASAPI 独占，减少系统混音/重采样
# extra = None
# try:
#     extra = sd.WasapiSettings(exclusive=True, category=sd.WasapiCategory.Capture)
# except Exception:
#     extra = None  # 设备不支持独占就用共享模式
#
# print("Recording...")
# x = sd.rec(int(seconds*fs),
#            samplerate=fs, channels=1, dtype='float32',
#            device=device, blocking=True, extra_settings=extra)
# np.save("test.npy", x)
# print("Done → rec.npy", x.shape, "fs=", fs)