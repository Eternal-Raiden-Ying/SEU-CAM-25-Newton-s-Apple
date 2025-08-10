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
from utils import phase_unwrap, fitting_line, normalize
from utils import (generate_chirp, get_bits_from_file, serial_to_parallel,
                   QPSK_mapping, OFDM_modulate)


project_dir = r"D:\Documents\Coding\Python\SEUCAM"
output_dir = os.path.join(project_dir,"Channel Measurement/output")
record_dir = os.path.join(project_dir,"Channel Measurement/record")

def print_stats(s, tag=""):
    re = s.real; im = s.imag
    print(f"--- {tag} ---")
    print("样本数:", s.size)
    print("实部 mean/std:", np.mean(re), np.std(re))
    print("虚部 mean/std:", np.mean(im), np.std(im))
    print("实/虚 std ratio:", np.std(re)/(np.std(im)+1e-12))
    print("实虚相关coef:", np.corrcoef(re, im)[0,1])
    print("平均幅度:", np.mean(np.abs(s)))
    print("幅度最小/最大:", np.min(np.abs(s)), np.max(np.abs(s)))
    print()

def remove_outliers_by_mad(s, k=6):
    """
    用基于MAD的方法检测幅度离群点并进行替换（保留相位，幅度限制到thresh）
    k: 阈值倍数，越大越保守
    """
    amp = np.abs(s)
    med = np.median(amp)
    mad = np.median(np.abs(amp - med)) + 1e-12
    thresh = med + k * mad
    # 只处理超出阈值的点，把其幅度缩到阈值，但保留原相位
    if np.any(amp > thresh):
        idx = amp > thresh
        phases = np.angle(s[idx])
        s_clean = s.copy()
        s_clean[idx] = thresh * np.exp(1j * phases)
        return s_clean, thresh, med, mad
    else:
        return s.copy(), thresh, med, mad

def iq_gain_phase_correction(s):
    """
    去均值 -> 幅度校准（使 std(real)==std(imag)）-> 用协方差白化去除相关与旋转
    返回校正后的符号
    """
    s0 = s.copy()
    # 1. 去均值
    mean_re = np.mean(s0.real)
    mean_im = np.mean(s0.imag)
    s1 = s0 - (mean_re + 1j*mean_im)
    # 2. 幅度校准（scale imag）
    re_std = np.std(s1.real)
    im_std = np.std(s1.imag)
    if im_std > 1e-12:
        s2 = s1.real + 1j * (s1.imag * (re_std / im_std))
    else:
        s2 = s1
    # 3. 去相关（2x2 协方差白化）
    # 将符号看成 2xN 实向量
    X = np.vstack([s2.real, s2.imag])
    C = np.cov(X)  # 2x2 实协方差矩阵
    # Eigen-decomposition for symmetric positive-definite-ish matrix
    w, V = np.linalg.eigh(C)
    # 防止数值问题
    w = np.clip(w, 1e-12, None)
    # 计算 C^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(w))
    W = D_inv_sqrt @ V.T  # whitening transform (real 2x2)
    X_white = W @ X
    s_white = X_white[0, :] + 1j * X_white[1, :]
    # 最后统一缩放到原始平均幅度
    scale = np.mean(np.abs(s2)) / (np.mean(np.abs(s_white)) + 1e-12)
    s_out = s_white * scale
    return s_out


def clean_and_fix_constellation(raw_constellation, mad_k=6, plot=True):
    s = raw_constellation.copy()
    print_stats(s, "原始")
    # 1. 去除极端幅值（用MAD）
    s_clean, thresh, med, mad = remove_outliers_by_mad(s, k=mad_k)
    print(f"MAD阈值: {thresh:.4f}, 中位数:{med:.4f}, MAD:{mad:.4f}")
    print_stats(s_clean, "去极端值后")
    # 2. IQ gain & phase correction + 去相关白化
    s_fixed = iq_gain_phase_correction(s_clean)
    print_stats(s_fixed, "校正后")
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.scatter(s.real, s.imag, s=2);
        plt.title("原始");
        plt.axhline(0, color='k');
        plt.axvline(0, color='k');
        plt.grid(True)
        plt.subplot(1, 3, 2)
        plt.scatter(s_clean.real, s_clean.imag, s=2);
        plt.title("去极端值后");
        plt.axhline(0, color='k');
        plt.axvline(0, color='k');
        plt.grid(True)
        plt.subplot(1, 3, 3)
        plt.scatter(s_fixed.real, s_fixed.imag, s=2);
        plt.title("校正后");
        plt.axhline(0, color='k');
        plt.axvline(0, color='k');
        plt.grid(True)
        plt.suptitle("Constellation: raw -> outlier removed -> corrected")
        plt.show()
    return s_fixed

def inspect_constellation(received_symbols, title="Constellation inspect"):
    s = received_symbols.flatten()
    re = s.real
    im = s.imag

    print("样本数:", s.size)
    print("实部 mean/std:", np.mean(re), np.std(re))
    print("虚部 mean/std:", np.mean(im), np.std(im))
    print("实/虚 std ratio:", np.std(re)/ (np.std(im)+1e-12))
    print("实虚相关coef:", np.corrcoef(re, im)[0,1])
    print("平均幅度:", np.mean(np.abs(s)))
    print("幅度最小/最大:", np.min(np.abs(s)), np.max(np.abs(s)))

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(re, im, s=2, alpha=0.6)
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.title(title)
    plt.xlabel('Real'); plt.ylabel('Imag')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.hist(np.abs(s), bins=80)
    plt.title("Amplitude histogram")
    plt.xlabel('|sym|')
    plt.grid(True)
    plt.show()







def get_top_idx_val(data: np.ndarray, n=5):
    data = data.flatten()
    n = int(n)
    assert data.size > n, "data.size should greater than n"
    idx = np.argpartition(data, -n)[-n:]
    sorted_idx = idx[np.argsort(-data[idx])]
    top_vals = data[sorted_idx]
    return sorted_idx, top_vals


def analysis_simple_pilots():
    fs = 48000
    N = 4096
    cp_len = 1024
    num_symbols = 100
    symbol_len = N + cp_len


    rx = np.load(fr"{record_dir}\100symbols\received_signal_diff_chirp_l2_10_24k_N4096_cp1024.npy")
    pilot = np.load(fr"{record_dir}\100symbols\pilot100_diff_cp1024.npy")
    chirp_template = generate_chirp(fs, duration=2, f_l=10, f_h=24000)

    corr = correlate(rx, chirp_template, mode='full')
    plt.plot(correlate(rx, chirp_template, mode='full'), color='blue',alpha=0.5, label='chirp_front')
    plt.show()
    ofdm_start = np.argmax(corr) + 1


    # check_radius = N + cp_len
    # corr = corr[:corr.size//100*100].reshape((-1,100))
    # corr_block = np.max(corr, axis=-1)
    # corr_block_idx = 100*np.arange(corr.shape[0])+np.argmax(corr, axis=-1)
    # corr_top5_idx, corr_top5_vals = get_top_idx_val(corr_block, 10)
    # corr_top5_idx = corr_block_idx[corr_top5_idx] - check_radius
    # for idx in corr_top5_idx:
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
    #     for index, rx in enumerate(rx_freq):
    #         H_f = rx /pilot
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


    # fine tune
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
    # fig, axes = plt.subplots(1,num_symbols, figsize=(5*num_symbols, 4))
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

    effective_symbols_num = num_symbols - 4  # in case that last symbols received have declined amplitude

    # demodulate and decode OFDM
    # get the average H(f)
    symbols = get_symbols(record=np.array(rx),cp_len=cp_len, N=N)
    H_f = evaluate_H_f(known_symbols=symbols[:min(5, effective_symbols_num),:], pilot_signals=pilot[:min(5, effective_symbols_num),:])

    # analysis impulse response
    h_t = np.fft.ifft(H_f)
    fig, axes = plt.subplots(1,2, figsize=(16,6))
    draw_in_TD(h_t.size/fs, h_t, title='Impluse response in time domain', ax=axes[0], x_label='time/s', y_label='h(t)')
    draw_in_FD(fs, h_t, title='Impluse response in freq domain', half=False, ax=axes[1],mode='Amplitude', y_label='H(f)/dB',x_label='Freq/Hz')
    plt.show()

    symbols = get_symbols(record=np.array(rx), cp_len=cp_len, N=N)
    H_fs = list()
    for i in range(effective_symbols_num):
        H_f = evaluate_H_f(known_symbols=symbols[i, :], pilot_signals=pilot[i])
        H_fs.append(H_f)

    # #  use the effective part(amp in time domain that doesn't decline)
    H_fs = H_fs[0:effective_symbols_num]
    # 逐差法，前后分两组
    H_f_former = H_fs[:effective_symbols_num//2]
    H_f_latter = H_fs[effective_symbols_num//2:effective_symbols_num//2*2]
    phase_shift = np.mean(np.stack(H_f_latter, axis=0) / np.stack(H_f_former, axis=0), axis=0)

    # # here are the code for testing an artificial freq offset
    # delta = (symbol_len/(fs+freq_bias)*fs - symbol_len)/symbol_len
    # print(f"sampling point delay about {int(delta*symbol_len)} points after a symbol")
    # ideal_phase_shift_unwrapped = np.arange(N)*np.floor(delta*8*symbol_len)*(2*np.pi)/N
    # plt.plot(ideal_phase_shift_unwrapped, linestyle='dotted', label='ideal')


    # # noted that there are some points where amplitude of phase shift far more than 1, which is unreasonable,
    # # so we would first exclude them, the reason why this occurs still remain unknown
    # # to see the amplitude of phase shift, run the code below
    # plt.title("Amplitude of H_f[i+8]/H_f[i]")
    # plt.xlabel("sub carrier wave frequency")
    # plt.ylabel("Amplitude")
    # plt.plot(np.abs(phase_shift))
    # plt.show()

    x_d = np.arange(phase_shift.size)
    default_unwrapped_phase = np.unwrap(np.angle(phase_shift), discont=1.5*np.pi)
    x_s, simple_unwrapped_phase = phase_unwrap(data=phase_shift, estimate_percent=0.1,
                                               initial_discont=1.5*np.pi, mode='simple')
    x_f, filtered_unwrapped_phase = phase_unwrap(data=phase_shift, estimate_percent=0.1,
                                                 initial_discont=1.5*np.pi, mode='filtered',
                                                 amp_filter_th=1.2)
    x_t, truncated_unwrapped_phase = phase_unwrap(data=phase_shift, estimate_percent=0.1,
                                                  initial_discont=1.5*np.pi, mode='truncated',
                                                  amp_filter_th=1.2)

    slope, intercept = fitting_line(x=x_t, y=truncated_unwrapped_phase, filter=True, residual_th=2)
    delta = slope / (effective_symbols_num//2*symbol_len*2*np.pi/N)
    freq_bias = fs/(delta + 1) - fs
    print(f"delta:{delta}")
    print(f"fs of receiver - fs of emitter = {freq_bias}")
    print(f"delay {delta * effective_symbols_num * symbol_len} points "
          f"after {effective_symbols_num} symbols(N:{N} and cp:{cp_len})")

    plt.title("Unwrap phase with different methods")
    plt.plot(slope*np.arange(N)+intercept, linestyle='solid', label='fitting result', color='red')
    plt.plot(slope*np.arange(N)+intercept+np.pi, linestyle='dotted', color='red', alpha=0.5)
    plt.plot(slope*np.arange(N)+intercept-np.pi, linestyle='dotted', color='red', alpha=0.5)
    plt.scatter(x_s,simple_unwrapped_phase,label='simple', s=1, color='royalblue', alpha=0.5)
    plt.scatter(x_f,filtered_unwrapped_phase,label='filtered', s=1, color='wheat', alpha=0.5)
    plt.scatter(x_t,truncated_unwrapped_phase, label='truncated', s=1, color='black', marker='*', alpha=0.5)
    plt.xlabel("sampling point")
    plt.ylabel("unwrapped phase")
    plt.legend()
    plt.show()

    # draw constellation distribution map
    n_rows = 4
    n_cols = 4
    pic_num = n_rows * n_cols
    pic_idx = np.linspace(start=1,
                          stop=1+(effective_symbols_num-1)//(pic_num-1)*(pic_num-1),
                          num=pic_num).astype(np.int32)

    fig, axes = plt.subplots(n_rows,n_cols, figsize=(10,12))
    for i, index in enumerate(pic_idx):
        raw_constellation = get_constellation(symbols=symbols[index, :], H_f=H_fs[0],
                                              approximation=non_approximate,
                                              symbol_len=N)
        ax = axes[i//n_cols,i%n_cols]
        draw_constellation_map(received=raw_constellation, emit_pilot=pilot[index,1:N//2],ax=ax,title=f"constellation{index+1}")
    fig.suptitle("original constellation")
    plt.tight_layout()
    plt.show()

    origin_H_f = H_fs[0] * np.exp(-1j*2*np.pi/N*intercept*np.arange(N))
    fig, axes = plt.subplots(n_rows,n_cols, figsize=(10,12))
    for i, index in enumerate(pic_idx):
        corrected_H_f = origin_H_f / np.exp(-1j*2*np.pi/N*(delta*index*symbol_len+intercept)*np.arange(N))
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=non_approximate,
                                                    symbol_len=N)
        ax = axes[i//n_cols,i%n_cols]
        draw_constellation_map(received=corrected_constellation, emit_pilot=pilot[index, 1:N//2],ax=ax,title=f"constellation{index+1}")
    fig.suptitle("corrected constellation")
    plt.tight_layout()
    plt.show()

    # statistical BER
    received_bits = list()
    emit_bits = QPSK_reflection(data=pilot[:effective_symbols_num, 1:N // 2]).flatten()
    for index in range(effective_symbols_num):
        raw_constellation = get_constellation(symbols=symbols[index, :], H_f=H_fs[0],
                                              approximation=simple_approximate,
                                              symbol_len=N)
        received_bits.append(QPSK_reflection(data=raw_constellation))
    received_bits = np.array(received_bits).flatten()
    BER = 1 - np.sum(np.equal(received_bits, emit_bits)) / received_bits.size
    print(f"bit error rate (uncorrected):{BER * 100:.4f}%")

    received_bits = list()
    for index in range(effective_symbols_num):
        corrected_H_f = origin_H_f / np.exp(-1j * 2 * np.pi / N * (delta * index * symbol_len + intercept) * np.arange(N))
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=simple_approximate,
                                                    symbol_len=N)
        received_bits.append(QPSK_reflection(data=corrected_constellation))
    received_bits = np.array(received_bits).flatten()
    BER = 1 - np.sum(np.equal(received_bits, emit_bits)) / received_bits.size
    print(f"bit error rate (corrected):{BER * 100:.4f}%")


def test():
    fs = 48000
    N = 4096
    cp_len = 1024
    num_symbols = 8
    symbol_len = N + cp_len

    rx = np.load(fr"{record_dir}\txt\received_signal_shakespeare_chirp_l2_10_24k_S8diff_N4096_cp1024_distortion.npy")
    pilot = np.load(fr"{record_dir}\txt\pilot_diff_txt_distortion.npy")
    chirp_template = generate_chirp(fs, duration=2, f_l=10, f_h=24000)

    corr = correlate(rx, chirp_template, mode='full')
    plt.plot(correlate(rx, chirp_template, mode='full'), color='blue', alpha=0.5, label='chirp_front')
    plt.show()
    ofdm_start = np.argmax(corr) + 1

    # fine tune
    # ofdm_start = ofdm_start + 10

    chirp_start = ofdm_start - chirp_template.size

    print(f"ofdm start {ofdm_start}")
    print(f"corr peak at {chirp_start}")

    print(f"开始提取 OFDM 符号")
    rx_pilot = rx[ofdm_start: ofdm_start + num_symbols * (N + cp_len)]

    #  plot OFDM parts respectively
    # fig, axes = plt.subplots(1,num_symbols, figsize=(5*num_symbols, 4))
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
    symbols = get_symbols(record=np.array(rx_pilot), cp_len=cp_len, N=N)
    H_f = evaluate_H_f(known_symbols=symbols[:min(5, effective_symbols_num), :],
                       pilot_signals=pilot[:min(5, effective_symbols_num), :])

    # analysis impulse response
    h_t = np.fft.ifft(H_f)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    draw_in_TD(h_t.size / fs, h_t, title='Impluse response in time domain', ax=axes[0], x_label='time/s',
               y_label='h(t)')
    draw_in_FD(fs, h_t, title='Impluse response in freq domain', half=False, ax=axes[1], mode='Amplitude',
               y_label='H(f)/dB', x_label='Freq/Hz')
    plt.show()

    symbols = get_symbols(record=np.array(rx_pilot), cp_len=cp_len, N=N)
    H_fs = list()
    for i in range(effective_symbols_num):
        H_f = evaluate_H_f(known_symbols=symbols[i, :], pilot_signals=pilot[i])
        H_fs.append(H_f)

    # #  use the effective part(amp in time domain that doesn't decline)
    H_fs = H_fs[0:effective_symbols_num]
    # 逐差法，前后分两组
    H_f_former = H_fs[:effective_symbols_num // 2]
    H_f_latter = H_fs[effective_symbols_num // 2:effective_symbols_num // 2 * 2]
    phase_shift = np.mean(np.stack(H_f_latter, axis=0) / np.stack(H_f_former, axis=0), axis=0)

    x_d = np.arange(phase_shift.size)
    default_unwrapped_phase = np.unwrap(np.angle(phase_shift), discont=1.5 * np.pi)
    x_s, simple_unwrapped_phase = phase_unwrap(data=phase_shift, estimate_percent=0.2,
                                               initial_discont=1.5 * np.pi, mode='simple')
    x_f, filtered_unwrapped_phase = phase_unwrap(data=phase_shift, estimate_percent=0.2,
                                                 initial_discont=1.5 * np.pi, mode='filtered',
                                                 amp_filter_th=1.2)
    x_t, truncated_unwrapped_phase = phase_unwrap(data=phase_shift, estimate_percent=0.1,
                                                  initial_discont=1.7 * np.pi, mode='truncated',
                                                  amp_filter_th=1.2, estimate_start=50)

    slope, intercept = fitting_line(x=x_t, y=truncated_unwrapped_phase, filter=True, residual_th=2)
    delta = slope / (effective_symbols_num // 2 * symbol_len * 2 * np.pi / N)
    freq_bias = fs / (delta + 1) - fs
    print(f"delta:{delta}")
    print(f"fs of receiver - fs of emitter = {freq_bias}")
    print(f"delay {delta * effective_symbols_num * symbol_len} points "
          f"after {effective_symbols_num} symbols(N:{N} and cp{cp_len})")

    plt.title("Unwrap phase with different methods")
    plt.plot(slope * np.arange(N) + intercept, linestyle='solid', label='fitting result', color='red')
    plt.plot(slope * np.arange(N) + intercept + np.pi, linestyle='dotted', color='red', alpha=0.5)
    plt.plot(slope * np.arange(N) + intercept - np.pi, linestyle='dotted', color='red', alpha=0.5)
    plt.scatter(x_s, simple_unwrapped_phase, label='simple', s=1, color='royalblue', alpha=0.5)
    plt.scatter(x_f, filtered_unwrapped_phase, label='filtered', s=1, color='wheat', alpha=0.5)
    plt.scatter(x_t, truncated_unwrapped_phase, label='truncated', s=1, color='black', marker='*', alpha=0.5)
    plt.xlabel("sampling point")
    plt.ylabel("unwrapped phase")
    plt.legend()
    plt.show()

    # draw constellation distribution map
    n_rows = 2
    n_cols = 4
    pic_num = n_rows * n_cols
    pic_idx = np.linspace(start=0,
                          stop=0+effective_symbols_num//(pic_num-1)*(pic_num-1),
                          num=pic_num).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    for i, index in enumerate(pic_idx):
        raw_constellation = get_constellation(symbols=symbols[index, :], H_f=H_fs[0],
                                              approximation=non_approximate,
                                              symbol_len=N)
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=raw_constellation, emit_pilot=pilot[index, 1:N // 2], ax=ax,
                               title=f"constellation{index + 1}")
    fig.suptitle("original constellation")
    plt.tight_layout()
    plt.show()

    origin_H_f = H_fs[0] * np.exp(-1j * 2 * np.pi / N * intercept * np.arange(N))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    for i, index in enumerate(pic_idx):
        corrected_H_f = origin_H_f / np.exp(-1j * 2 * np.pi / N * (delta * index * symbol_len + intercept) * np.arange(N))
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=non_approximate,
                                                    symbol_len=N)
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=corrected_constellation, emit_pilot=pilot[index, 1:N // 2], ax=ax,
                               title=f"constellation{index + 1}")
    fig.suptitle("corrected constellation")
    plt.tight_layout()
    plt.show()

    # statistical BER
    origin_H_f = H_fs[0] * np.exp(-1j * 2 * np.pi / N * intercept * np.arange(N))
    received_bits = list()
    emit_bits = QPSK_reflection(data=pilot[:effective_symbols_num, 1:N // 2]).flatten()
    for index in range(effective_symbols_num):
        raw_constellation = get_constellation(symbols=symbols[index, :], H_f=H_fs[0],
                                              approximation=simple_approximate,
                                              symbol_len=N)
        received_bits.append(QPSK_reflection(data=raw_constellation))
    received_bits = np.array(received_bits).flatten()
    BER = 1 - np.sum(np.equal(received_bits, emit_bits)) / received_bits.size
    print(f"bit error rate (uncorrected):{BER * 100:.4f}%")

    received_bits = list()
    for index in range(effective_symbols_num):
        corrected_H_f = origin_H_f / np.exp(-1j * 2 * np.pi / N * (delta * index * symbol_len + intercept) * np.arange(N))
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=simple_approximate,
                                                    symbol_len=N)
        received_bits.append(QPSK_reflection(data=corrected_constellation))
    received_bits = np.array(received_bits).flatten()
    BER = 1 - np.sum(np.equal(received_bits, emit_bits)) / received_bits.size
    print(f"bit error rate (corrected):{BER * 100:.4f}%")

    print("Start to extract data part")
    delay = np.round(delta * effective_symbols_num * symbol_len).astype(np.int32)
    plt.plot(rx)
    plt.axvline(ofdm_start, linestyle='dotted', color='red')
    plt.axvline(ofdm_start + (num_symbols+32) * (N+cp_len)+delay, linestyle='dotted', color='red')
    plt.axvline(ofdm_start + num_symbols * (N + cp_len) + delay, linestyle='dotted', color='red')
    plt.show()

    rx_data = rx[ofdm_start + num_symbols * (N + cp_len) + delay:]
    symbols = get_symbols(record=rx_data, N=N, cp_len=cp_len)
    received_bits = list()
    emit_bits = get_bits_from_file(r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\data\shakespace(short).txt")
    emit_constellations = QPSK_mapping(serial_to_parallel(emit_bits, N=N))
    n_rows = 4
    n_cols = 4
    pic_num = n_rows * n_cols
    pic_idx = np.linspace(start=0,
                          stop=0 + emit_constellations.shape[0] // (pic_num - 1) * (pic_num - 1),
                          num=pic_num).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))
    for i, index in enumerate(pic_idx):
        corrected_H_f = origin_H_f / np.exp(-1j * 2 * np.pi / N * (delta * (index) * symbol_len + intercept) * np.arange(N))
        constellation = get_constellation(symbols=symbols[index], H_f=corrected_H_f,
                                              approximation=non_approximate,
                                              symbol_len=N)
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=constellation, emit_pilot=emit_constellations[index], ax=ax,
                               title=f"constellation{index + 1}")
    fig.suptitle("data constellation")
    plt.tight_layout()
    plt.show()


    for index, symbol in enumerate(symbols):
        corrected_H_f = origin_H_f / np.exp(-1j*2*np.pi/N*(delta*(num_symbols+index)*symbol_len+intercept)*np.arange(N))
        constellation = get_constellation(symbols=symbol, H_f=corrected_H_f,
                                          approximation=simple_approximate)
        received_bits.append(QPSK_reflection(data=constellation,judge_radius=np.sqrt(2)/2))

    received_bits = np.array(received_bits).flatten()
    received_bits = received_bits[:emit_bits.size]
    print(symbols.shape[0])

    emit_bit0 = emit_bits.reshape(-1,2)[:,1]
    received_bit0 = received_bits.reshape(-1,2)[:,1]
    print(1-np.sum(np.equal(emit_bit0.flatten(), received_bit0.flatten()))/emit_bit0.size)

    emit_bit1 = emit_bits.reshape(-1,2)[:,0]
    received_bit1 = received_bits.reshape(-1,2)[:,0]
    print(1-np.sum(np.equal(emit_bit1.flatten(), received_bit1.flatten()))/emit_bit0.size)

    cheat_bits = np.concatenate([np.zeros((emit_bits.size//8,1)),
                                 received_bits.reshape(-1,8)[:,1:8]], axis=1).flatten().astype(np.uint8)

    #bytes = get_bytes(np.array(cheat_bits).flatten(), bitorder='little')
    bytes = np.packbits(cheat_bits)
    with open("shakespeare.txt", 'wb') as file:
        file.write(bytes.tobytes())


if __name__ == "__main__":
    test()




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