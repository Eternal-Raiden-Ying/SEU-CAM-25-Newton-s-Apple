import sys
import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import correlate, chirp

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import (get_symbols, get_constellation, simple_approximate,
                   QPSK_reflection,get_bytes, evaluate_H_f, non_approximate)
from utils import draw_in_TD, draw_in_FD, draw_constellation_map
from utils import decode_bytes


def generate_chirp(fs, duration=1, f0=10, f1=24000):
    t = np.linspace(0, duration, int(fs * duration))
    chirp_sig = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    return chirp_sig


output_dir = r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\output\OFDM"


if __name__ == "__main__":
    # 参数一致
    fs = 48000
    N = 4096
    cp_len = 256
    num_symbols = 5
    symbol_len = N + cp_len
    total_len = num_symbols * symbol_len

    rx = np.load(r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\ofdm_signal_chirp_l2_f24k_fs48k_N4096_S5.npy")
    pilot = np.load(r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\pilot_symbol_in_freq.npy")

    chirp_template = generate_chirp(fs,duration=2,f0=10,f1=24000)


    corr = correlate(rx,chirp_template, mode='full')

    ofdm_start = np.argmax(np.abs(corr)) + 1

    # # fine tune
    # ofdm_start = ofdm_start + 10

    chirp_start = ofdm_start - chirp_template.size

    print(f"ofdm start {ofdm_start}")
    print(f"corr peak at {chirp_start}")

    # fig, axes = plt.subplots(3,1)
    # axes[0].plot(corr)
    # axes[0].set_xlim(0,250000)
    # axes[0].set_title("corr")
    # axes[1].plot(rx)
    # axes[1].axvline(x=ofdm_start,color='r')
    # axes[1].axvline(x=chirp_start, color='g')
    # axes[1].set_xlim(0,250000)
    # axes[1].set_title("received")
    #
    #
    # axes[2].plot(np.concatenate([np.zeros(chirp_start),chirp_template]))
    # axes[2].set_title("chirp (after shifted)")
    # axes[2].set_xlim(0,250000)
    #
    # plt.show()


    print(f"开始提取 OFDM 符号")
    zoom = rx[ofdm_start-2*(N+cp_len) : ofdm_start + num_symbols * (N + cp_len)]
    draw_in_TD(zoom.size/fs, zoom,title='zoom rx')
    plt.show()

    rx = rx[ofdm_start : ofdm_start + num_symbols * (N + cp_len)]
    fig, axes = plt.subplots(1,5,figsize=(25,4))
    for i in range(num_symbols):
        draw_in_TD(time=symbol_len/fs,
                   signal=rx[i*symbol_len:(i+1)*symbol_len],
                   title=f'symbol {i+1}',
                   x_label='time/s',
                   y_label='received_signal',
                   ax=axes[i])

    fig.tight_layout()
    plt.show()



    # symbols = get_symbols(record=np.array(rx), cp_len=cp_len, N=N)
    # H_fs = list()
    # for i in range(num_symbols):
    #     H_f = evaluate_H_f(known_symbols=symbols[i,:], pilot_signals=pilot)
    #     h_t = np.fft.ifft(H_f)
    #     fig, axes = plt.subplots(1,2, figsize=(16,6))
    #     draw_in_TD(h_t.size/fs, h_t, title=f"Impulse response evaluated from {i+1} symbol",
    #                x_label="time/s", y_label="h(t)", ax=axes[0])
    #     draw_in_FD(fs, H_f, title=f"Impulse response evaluated from {i+1} symbol",
    #                x_label="freq/Hz", y_label="H(f)/dB", ax=axes[1])
    #     plt.show()



    # demodulate and decode OFDM symbols
    symbols = get_symbols(record=np.array(rx),cp_len=cp_len, N=N)
    H_f = evaluate_H_f(known_symbols=symbols[:3,:], pilot_signals=pilot)
    constellations = get_constellation(symbols=symbols[2:,:], H_f=H_f,
                                       approximation=simple_approximate,
                                       symbol_len=N)
    bits = QPSK_reflection(constellations)
    byte_data = get_bytes(bits)

    # decode_bytes(byte_data=byte_data, pth=output_dir)


    # analysis impulse response
    h_t = np.fft.ifft(H_f)
    fig, axes = plt.subplots(1,2, figsize=(16,6))
    draw_in_TD(h_t.size/fs, h_t, title='Impluse response in time domain', ax=axes[0], x_label='time/s', y_label='h(t)')
    draw_in_FD(fs, h_t, title='Impluse response in freq domain', half=True, ax=axes[1],mode='Amplitude', y_label='H(f)/dB',x_label='Freq/Hz')
    plt.show()


    # draw constellation distribution map
    raw_constellation = get_constellation(symbols=symbols[2:,:], H_f=H_f,
                                          approximation=non_approximate,
                                          symbol_len=N)

    draw_constellation_map(received=raw_constellation, emit_pilot=pilot[1:N//2])



