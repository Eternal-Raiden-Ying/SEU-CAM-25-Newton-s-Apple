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

project_dir = r"D:\Documents\Coding\Python\SEUCAM"
output_dir = os.path.join(project_dir,"Channel Measurement/output")
record_dir = os.path.join(project_dir,"Channel Measurement/record")


def get_top_idx_val(data:np.ndarray, n=5):
    data = data.flatten()
    n = int(n)
    assert data.size > n, "data.size should greater than n"
    idx = np.argpartition(data, -n)[-n:]
    sorted_idx = idx[np.argsort(-data[idx])]
    top_vals = data[sorted_idx]
    return sorted_idx, top_vals


def analysis():
    # 参数一致
    fs = 48000
    N = 4096
    cp_len = 2000
    num_symbols = 16
    symbol_len = N + cp_len
    total_len = num_symbols * symbol_len

    freq_bias = 2.583498677248677e-04  # actual_freq - ideal_freq


    rx = np.load(fr"{record_dir}\ofdm_signal_l2_10_18k_cp2k_fs_48k_N4096_S16_l2_20_24k\received.npy")
    pilot = np.load(fr"{record_dir}\ofdm_signal_l2_10_18k_cp2k_fs_48k_N4096_S16_l2_20_24k\pilot.npy")
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
    # fig, axes = plt.subplots(1,num_symbols)
    # for i in range(num_symbols):
    #     draw_in_TD(time=symbol_len/fs,
    #                signal=rx[i*symbol_len:(i+1)*symbol_len],
    #                title=f'symbol {i+1}',
    #                x_label='time/s',
    #                y_label='received_signal',
    #                ax=axes[i])
    #
    # fig.tight_layout()
    # plt.show()

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
    # plt.title("Amplitude of H_f[i+8]/H_f[i]")
    # plt.xlabel("sub carrier wave frequency")
    # plt.ylabel("Amplitude")
    # plt.plot(np.abs(phase_shift))
    # plt.show()

    x = np.arange(phase_shift.size)
    actual_phase_shift_unwrapped = phase_unwrap(np.angle(phase_shift), estimate_percent=0.2)
    # plt.plot(x, np.unwrap(np.angle(phase_shift), discont= 1.5*np.pi), label='default')
    # plt.plot(x, actual_phase_shift_unwrapped,label='improved unwrap')
    # plt.legend()
    # plt.xlabel("sub carrier wave freq")
    # plt.show()


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
    received_bits = list()
    fig, axes = plt.subplots(n_rows,n_cols, figsize=(10,12))
    for index in range(effective_symbols_num):
        corrected_H_f = origin_H_f / np.exp(-1j*2*np.pi/N*(delta*index*symbol_len+intercept)*np.arange(N))
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=non_approximate,
                                                    symbol_len=N)
        received_bits.append(get_bytes(QPSK_reflection(corrected_constellation)))
        ax = axes[index//n_cols,index%n_cols]
        draw_constellation_map(received=corrected_constellation, emit_pilot=pilot[1:N//2],ax=ax,title=f"constellation{index+1}")
    fig.suptitle("corrected constellation")
    plt.tight_layout()
    plt.show()



def test():
    fs = 48000  # 采样率
    N = 4096  # OFDM 子载波数
    cp_len = 128  # 循环前缀长度
    symbol_len = cp_len + N

    pilot_num = 8

    chirp_len = 1
    chirp_fl1 = 10
    chirp_fl2 = 20
    chirp_fh1 = 18000
    chirp_fh2 = 24000

    rx = np.load(r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\shakespare\blockpilot_shakespace_4.npy")
    pilot = np.load(r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\shakespare\pilot_signal(short).npy")
    # synchronize
    chirp_front = generate_chirp(fs=fs, duration=chirp_len, f_l=chirp_fl1, f_h=chirp_fh1)
    chirp_back = generate_chirp(fs=fs, duration=chirp_len, f_l=chirp_fl2, f_h=chirp_fh2)
    pilot_start = np.argmax(correlate(rx, chirp_front, mode='full'))+1
    ofdm_start = np.argmax(correlate(rx, chirp_back, mode='full'))+1
    ofdm_start = ofdm_start
    print(f"pilot_start:{pilot_start}, ofdm_start:{ofdm_start}")
    plt.plot(correlate(rx, chirp_front, mode='full'), color='blue',alpha=0.5, label='chirp_front')
    plt.plot(correlate(rx, chirp_back, mode='full'), color='orange', alpha=0.5, label='chirp_back')
    plt.legend()
    plt.show()
    # cp_corr = list()
    # check_radius = int(N*1.5)
    # for i in range(-check_radius,check_radius):
    #     cp_start = ofdm_start + i
    #     prefix = rx[cp_start: cp_start+cp_len]
    #     symbol_part = rx[cp_start+N: cp_start+cp_len+N]
    #     cp_corr.append(correlate(prefix,symbol_part, mode='valid'))
    # cp_corr = np.array(cp_corr).flatten()
    # plt.plot(np.linspace(-check_radius, check_radius, 2*check_radius), cp_corr)
    # plt.show()
    # cp_corr = cp_corr[:cp_corr.size//100*100].reshape((-1,100))
    # cp_corr_block = np.max(cp_corr, axis=-1)
    # cp_corr_block_idx = 100*np.arange(cp_corr.shape[0])+np.argmax(cp_corr, axis=-1)
    # cp_corr_top5_idx, cp_corr_top5_vals = get_top_idx_val(cp_corr_block, 10)
    # cp_corr_top5_idx = cp_corr_block_idx[cp_corr_top5_idx] - check_radius
    # for idx in cp_corr_top5_idx:
    #     rx_syn = rx[ofdm_start + idx:]
    #
    #     # extract symbols
    #     symbol_len = N+cp_len
    #     symbol_num = rx_syn.size // symbol_len
    #     rx_syn = rx_syn[:symbol_num*symbol_len].reshape((symbol_num, symbol_len))[:,cp_len:]
    #     rx_freq = np.fft.fft(rx_syn, axis=1)
    #
    #     # estimate H(f)
    #     H_fs = list()
    #     for index, known_pilot in enumerate(known_pilots):
    #         H_f = rx_freq[index] / known_pilot
    #         H_fs.append(H_f)
    #
    #     H_f = np.mean(np.array(H_fs), axis=0)
    #     h_t = np.fft.ifft(H_f)
    #     fig, axes = plt.subplots(1,2, figsize=(10,6))
    #     draw_in_FD(fs, h_t, ax=axes[1])
    #     draw_in_TD(h_t.size/fs, h_t, ax=axes[0])
    #     fig.suptitle(f"idx: {idx}")
    #     fig.tight_layout()
    #     plt.show()

    # estimate H(f)
    rx_pilot = rx[pilot_start: pilot_start+pilot_num*(N+cp_len)]
    symbols = get_symbols(record=rx_pilot, cp_len=cp_len, N=N)
    H_fs = list()
    for idx in range(pilot_num):
        H_fs.append(evaluate_H_f(known_symbols=symbols[idx], pilot_signals=pilot))
    H_fs = np.array(H_fs)
    H_fs = H_fs[0:pilot_num]

    h_t = np.fft.ifft(np.mean(np.array(H_fs), axis=0))
    draw_in_TD(h_t.size/fs, h_t)
    draw_in_FD(fs, h_t)
    plt.show()

    # 逐差法，前后分两组
    discont = np.pi
    H_f_former = H_fs[:pilot_num // 2]
    H_f_latter = H_fs[pilot_num // 2:pilot_num // 2 * 2]
    phase_shift = np.mean(np.stack(H_f_latter, axis=0) / np.stack(H_f_former, axis=0), axis=0)
    phase_shift_truncated = np.mean(np.stack(H_f_latter, axis=0) / np.stack(H_f_former, axis=0), axis=0)[:N // 2]

    x = np.arange(phase_shift.size)
    actual_phase_shift_unwrapped = phase_unwrap(np.angle(phase_shift), estimate_percent=0.2, initial_discont=discont)

    mask_f = np.where(np.abs(phase_shift) < 1.2)
    x_f = x[mask_f]
    filtered_phase_shift = phase_shift[mask_f]
    filtered_phase_shift_unwrapped = phase_unwrap(np.angle(filtered_phase_shift), estimate_percent=0.2,
                                                  x_filtered=x_f, initial_discont=discont)

    x_truncate = np.arange(phase_shift_truncated.size)
    mask_trunc = np.where(np.abs(phase_shift_truncated) < 1.2)
    truncated_phase_shift = phase_shift_truncated[mask_trunc]
    truncated_phase_shift_unwrapped = phase_unwrap(np.angle(truncated_phase_shift), estimate_percent=0.2,
                                                   x_filtered=x_truncate[mask_trunc], initial_discont=discont)

    x1 = x_truncate[mask_trunc]
    coeffs = np.polyfit(x1, truncated_phase_shift_unwrapped, deg=1)
    phase_fit = np.polyval(coeffs, x1)
    residual = truncated_phase_shift_unwrapped - phase_fit
    std = np.std(residual)
    mask_res = np.where(np.abs(residual) < 2 * std)
    coeffs_refined = np.polyfit(x1[mask_res], truncated_phase_shift_unwrapped[mask_res], deg=1)
    slope, intercept = coeffs_refined
    delta = slope / (pilot_num // 2 * symbol_len * 2 * np.pi / N)
    print(f"delta:{delta}")
    freq_bias = fs / (delta + 1) - fs
    print(f"fs of receiver - fs of emitter = {freq_bias}")
    print(f"delay {delta * pilot_num * symbol_len} points after {pilot_num} symbols(N4096 and cp2000)")
    plt.title("Unwrap phase with different methods")
    plt.plot(slope * np.arange(N) + intercept, linestyle='solid', label='fitting result', color='red')
    plt.plot(slope * np.arange(N) + intercept+discont, linestyle='dotted', color='red', alpha=0.5)
    plt.plot(slope * np.arange(N) + intercept-discont, linestyle='dotted', color='red', alpha=0.5)
    plt.scatter(x, actual_phase_shift_unwrapped, label='simple', s=1, color='royalblue', alpha=0.5)
    plt.scatter(x_f, filtered_phase_shift_unwrapped, label='filtered', s=1, color='wheat', alpha=0.5)
    plt.scatter(x1[mask_res], truncated_phase_shift_unwrapped[mask_res], label='truncated', s=1, color='black',
                marker='*', alpha=0.5)
    plt.xlabel("sampling point")
    plt.ylabel("unwrapped phase")
    plt.legend()
    plt.show()

    raw_constellations_pilot = get_constellation(symbols=symbols, H_f=H_fs[0], approximation=non_approximate, symbol_len=N)
    n_rows = 2
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols)
    for index in range(pilot_num):
        ax = axes[index // n_cols, index % n_cols]
        draw_constellation_map(received=raw_constellations_pilot[index], emit_pilot=pilot[1:N // 2], ax=ax, title=f"constellation{index + 1}")
    fig.suptitle("original constellation")
    plt.tight_layout()
    plt.show()

    origin_H_f = H_fs[0] * np.exp(-1j * 2 * np.pi / N * intercept * np.arange(N))
    received_bits = list()
    fig, axes = plt.subplots(n_rows, n_cols)
    for index in range(pilot_num):
        corrected_H_f = origin_H_f / np.exp(-1j * 2 * np.pi / N * (delta * index * symbol_len + intercept) * np.arange(N))
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=non_approximate,
                                                    symbol_len=N)
        received_bits.append(get_bytes(QPSK_reflection(corrected_constellation)))
        ax = axes[index // n_cols, index % n_cols]
        draw_constellation_map(received=corrected_constellation, emit_pilot=pilot[1:N // 2], ax=ax,
                               title=f"constellation{index + 1}")
    fig.suptitle("corrected constellation")
    plt.tight_layout()
    plt.show()

    rx_syn = rx[ofdm_start:]
    # extract symbols
    symbol_len = N+cp_len
    symbols = get_symbols(record=rx_syn, cp_len=cp_len, N=N)
    H_f = evaluate_H_f(known_symbols=symbols[0:8], pilot_signals=known_pilots)
    raw_constellations = get_constellation(symbols=symbols, H_f=H_f,approximation=non_approximate,symbol_len=N)
    n_rows = 2
    n_cols = 4
    assert n_rows * n_cols == effective_symbols_num, "make sure subplots has right rows and cols"

    fig, axes = plt.subplots(n_rows,n_cols, figsize=(10,12))
    for index in range(effective_symbols_num):
        ax = axes[index//n_cols,index%n_cols]
        draw_constellation_map(received=raw_constellations[index], emit_pilot=known_pilots[index][1:N//2],ax=ax,title=f"constellation{index+1}")
    fig.suptitle("original constellation")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()
