import sys
import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import correlate, chirp

# below are relative import, ignore the warning, they won't influence the code
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import (get_symbols, get_constellation, simple_approximate,
                   QPSK_reflection,get_bytes, evaluate_H_f, non_approximate)
from utils import draw_in_TD, draw_in_FD, draw_constellation_map
from utils import decode_bytes
from utils import phase_unwrap
from utils import generate_chirp

output_dir = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\save"


def analysis():
    # 参数一致
    fs = 48000
    N = 4096
    cp_len = 2000
    num_symbols = 16
    symbol_len = N + cp_len
    total_len = num_symbols * symbol_len

    freq_bias = 2.583498677248677e-04  # actual_freq - ideal_freq


    rx = np.load(r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\ofdm_signal_l2_10_18k_cp2k_fs_48k_N4096_S16_l2_20_24k\received.npy")
    pilot = np.load(r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\ofdm_signal_l2_10_18k_cp2k_fs_48k_N4096_S16_l2_20_24k\pilot.npy")
    chirp_template = generate_chirp(fs, duration=2, f_l=10, f_h=18000)

    corr = correlate(rx, chirp_template, mode='full')

    ofdm_start = np.argmax(np.abs(corr)) + 1

    # # fine tune
    # ofdm_start = ofdm_start + 10

    chirp_start = ofdm_start - chirp_template.size

    print(f"ofdm start {ofdm_start}")
    print(f"corr peak at {chirp_start}")

    print(f"开始提取 OFDM 符号")
    # here are codes to plot zoom of the extract signal in time domain
    # zoom = rx[ofdm_start-symbol_len : ofdm_start + num_symbols * symbol_len+ symbol_len]
    # draw_in_TD(zoom.size/fs, zoom,title='zoom rx')
    # plt.show()
    rx = rx[ofdm_start: ofdm_start + num_symbols * (N + cp_len)]

    #  plot OFDM parts respectively
    fig, axes = plt.subplots(1,num_symbols)
    for i in range(num_symbols):
        draw_in_TD(time=symbol_len/fs,
                   signal=rx[i*symbol_len:(i+1)*symbol_len],
                   title=f'symbol {i+1}',
                   x_label='time/s',
                   y_label='received_signal',
                   ax=axes[i])

    fig.tight_layout()
    plt.show()

    effective_symbols_num = num_symbols

    # demodulate and decode OFDM
    # get the average H(f)
    symbols = get_symbols(record=np.array(rx),cp_len=cp_len, N=N)
    H_f = evaluate_H_f(known_symbols=symbols[:effective_symbols_num,:], pilot_signals=pilot)

    # analysis impulse response
    h_t = np.fft.ifft(H_f)
    fig, axes = plt.subplots(1,2, figsize=(16,6))
    draw_in_TD(h_t.size/fs, h_t, title='Impluse response in time domain', ax=axes[0], x_label='time/s', y_label='h(t)')
    draw_in_FD(fs, h_t, title='Impluse response in freq domain', half=False, ax=axes[1],mode='Amplitude', y_label='H(f)/dB',x_label='Freq/Hz')
    plt.show()

    symbols = get_symbols(record=np.array(rx), cp_len=cp_len, N=N)
    H_fs = list()
    for i in range(effective_symbols_num):
        H_f = evaluate_H_f(known_symbols=symbols[i, :], pilot_signals=pilot)
        H_fs.append(H_f)

    # #  use the effective part(amp in time domain that doesn't decline)
    H_fs = H_fs[0:effective_symbols_num]
    # 逐差法，前后分两组
    H_f_former = H_fs[:effective_symbols_num//2]
    H_f_latter = H_fs[effective_symbols_num//2:effective_symbols_num//2*2]
    phase_shift = np.mean(np.stack(H_f_latter, axis=0) / np.stack(H_f_former, axis=0), axis=0)
    phase_shift_truncated = np.mean(np.stack(H_f_latter, axis=0) / np.stack(H_f_former, axis=0), axis=0)[:N//2]

    # here are the code for testing an artificial freq offset
    # delta = (symbol_len/(fs+freq_bias)*fs - symbol_len)/symbol_len
    # print(f"sampling point delay about {int(delta*symbol_len)} points after a symbol")
    # ideal_phase_shift_unwrapped = np.arange(N)*np.floor(delta*8*symbol_len)*(2*np.pi)/N
    # plt.plot(ideal_phase_shift_unwrapped, linestyle='dotted', label='ideal')


    # noted that there are some points where amplitude of phase shift far more than 1, which is unreasonable,
    # so we would first exclude them, the reason why this occurs still remain unknown
    # to see the amplitude of phase shift, run the code below
    # plt.plot(np.abs(phase_shift))
    # plt.show()

    x = np.arange(phase_shift.size)
    actual_phase_shift_unwrapped = phase_unwrap(np.angle(phase_shift), estimate_percent=0.2)

    mask_f = np.where(np.abs(phase_shift) < 1.2)
    x_f = x[mask_f]
    filtered_phase_shift = phase_shift[mask_f]
    filtered_phase_shift_unwrapped = phase_unwrap(np.angle(filtered_phase_shift), estimate_percent=0.2, x_filtered=x_f)

    x_truncate = np.arange(phase_shift_truncated.size)
    mask_trunc = np.where(np.abs(phase_shift_truncated) < 1.2)
    truncated_phase_shift = phase_shift_truncated[mask_trunc]
    truncated_phase_shift_unwrapped = phase_unwrap(np.angle(truncated_phase_shift), estimate_percent=0.2, x_filtered=x_truncate[mask_trunc])

    x1 = x_truncate[mask_trunc]
    coeffs = np.polyfit(x1, truncated_phase_shift_unwrapped, deg=1)
    phase_fit = np.polyval(coeffs, x1)
    residual = truncated_phase_shift_unwrapped - phase_fit
    std = np.std(residual)
    mask_res = np.where(np.abs(residual) < 2*std)
    coeffs_refined = np.polyfit(x1[mask_res], truncated_phase_shift_unwrapped[mask_res], deg=1)
    slope, intercept = coeffs_refined
    delta = slope / (effective_symbols_num//2*symbol_len*2*np.pi/N)
    print(f"delta:{delta}")
    freq_bias = fs/(delta + 1) - fs
    print(f"fs of receiver - fs of emitter = {freq_bias}")
    print(f"delay {delta*effective_symbols_num*symbol_len} points after {effective_symbols_num} symbols(N4096 and cp2000)")
    plt.title("Unwrap phase with different methods")
    plt.plot(slope*np.arange(N)+intercept, linestyle='solid', label='fitting result', color='red')
    plt.scatter(x,actual_phase_shift_unwrapped,label='simple', s=1, color='royalblue', alpha=0.5)
    plt.scatter(x_f,filtered_phase_shift_unwrapped,label='filtered', s=1, color='wheat', alpha=0.5)
    plt.scatter(x1[mask_res],truncated_phase_shift_unwrapped[mask_res], label='truncated', s=1, color='black', marker='*', alpha=0.5)
    plt.xlabel("sampling point")
    plt.ylabel("unwrapped phase")
    plt.legend()
    plt.show()

    # draw constellation distribution map
    raw_constellation = get_constellation(symbols=symbols[:effective_symbols_num,:], H_f=H_fs[0],
                                          approximation=non_approximate,
                                          symbol_len=N)

    n_rows = 4
    n_cols = 4
    assert n_rows * n_cols == effective_symbols_num, "make sure subplots has right rows and cols"

    fig, axes = plt.subplots(n_rows,n_cols, figsize=(10,12))
    for index in range(effective_symbols_num):
        ax = axes[index//n_cols,index%n_cols]
        draw_constellation_map(received=raw_constellation[index], emit_pilot=pilot[1:N//2],ax=ax,title=f"constellation{index+1}")
    fig.suptitle("original constellation")
    plt.tight_layout()
    plt.show()

    origin_H_f = H_fs[0] * np.exp(-1j*2*np.pi/N*intercept*np.arange(N))
    fig, axes = plt.subplots(n_rows,n_cols, figsize=(10,12))
    for index in range(effective_symbols_num):
        corrected_H_f = origin_H_f / np.exp(-1j*2*np.pi/N*(delta*index*symbol_len+intercept)*np.arange(N))
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=non_approximate,
                                                    symbol_len=N)
        ax = axes[index//n_cols,index%n_cols]
        draw_constellation_map(received=corrected_constellation, emit_pilot=pilot[1:N//2],ax=ax,title=f"constellation{index+1}")
    fig.suptitle("corrected constellation")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analysis()
