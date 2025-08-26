import sys
import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import correlate, chirp, lfilter

# below are relative import, ignore the warning, they won't influence the code
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import (get_symbols, get_constellation, simple_approximate,
                   QPSK_reflection, get_bytes, evaluate_H_f, non_approximate, normalize_approximate)
from utils import draw_in_TD, draw_in_FD, draw_constellation_map
from utils import decode_bytes,descrambler
from utils import phase_unwrap, fitting_line, normalize, phase_unwrap_auto
from utils import ldpc_encode_bits,scrambler, scrambler_random
from utils import (generate_chirp, get_bits_from_file, serial_to_parallel,
                   QPSK_mapping, OFDM_modulate)
from utils import LDPC_PY_PATH

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

LDPC_STANDARD = '802.11n'
LDPC_RATE = '1/2'
LDPC_Z = 27
LDPC_PTYPE = 'A'  # only for 802.16 rate 2/3 or 3/4

project_dir = r"D:\Documents\Coding\Python\SEUCAM"
output_dir = os.path.join(project_dir, "Channel Measurement/output/ldpc")
record_dir = os.path.join(project_dir, "Channel Measurement/record")
data_dir = os.path.join(project_dir, "Channel Measurement/data")

def correct_H_f(origin_H_f, delta, N, index, symbol_len, fixed_phase_shift_factor=0):
    k = np.linspace(-N//2,N//2,N,endpoint=False,dtype=np.int32)
    k = np.concatenate([k[N//2:],k[:N//2]])
    corrected_H_f = origin_H_f * np.exp(1j * (-2 * np.pi / N * delta * index * symbol_len * k))
    return corrected_H_f

def caculate_constellation_std(symbols, origin_H_f, delta, fixed_phase_shift_factor, pilot, N, symbol_len, num_symbols,clockwise=False):
    """
        QPSK -> (b0,b1) coresponding to that in qpsk_llrs_from_constellation
    :param symbols:
    :param origin_H_f:
    :param delta:
    :param fixed_phase_shift_factor:
    :param pilot:
    :param N:
    :param symbol_len:
    :param num_symbols:
    :return:
    """
    b0_stds = []
    b1_stds = []
    for index in range(num_symbols):
        corrected_H_f = correct_H_f(origin_H_f, delta, N, index, symbol_len)
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=non_approximate,
                                                    symbol_len=N)
        emit_bits = QPSK_reflection(data=pilot[index, 1:N // 2],clockwise=clockwise).flatten()
        emit_b1 = emit_bits.reshape(-1,2) [:,1]
        emit_b0 = emit_bits.reshape(-1,2) [:,0]
        if clockwise:
            receive_b1 = corrected_constellation.imag.flatten()
            receive_b0 = corrected_constellation.real.flatten()
        else:
            receive_b1 = corrected_constellation.real.flatten()
            receive_b0 = corrected_constellation.imag.flatten()

        b1_std = (np.std(receive_b1[emit_b1==0]) + np.std(receive_b1[emit_b1==1]))/2
        b0_std = (np.std(receive_b0[emit_b0==0]) + np.std(receive_b0[emit_b0==1]))/2
        b0_stds.append(b0_std)
        b1_stds.append(b1_std)
    return b0_stds, b1_stds

def estimate_constellation_std(constellation, *, clockwise=False):
    if clockwise:
        receive_b1 = constellation.imag.flatten()
        receive_b0 = constellation.real.flatten()
    else:
        receive_b1 = constellation.real.flatten()
        receive_b0 = constellation.imag.flatten()
    receive_b0 = np.abs(receive_b0)
    receive_b1 = np.abs(receive_b1)
    b0_std = np.sqrt(np.sum(np.square(receive_b0 - np.sqrt(2)/2))/receive_b0.size)
    b1_std = np.sqrt(np.sum(np.square(receive_b1 - np.sqrt(2)/2))/receive_b1.size)
    # receive_b0_remote = receive_b0[receive_b0>np.sqrt(2)]
    # receive_b1_remote = receive_b1[receive_b1>np.sqrt(2)]
    # b0_std = np.sqrt(np.sum(np.square(np.concatenate([receive_b0_remote,receive_b0]) - np.sqrt(2)/2))/(receive_b0.size))
    # b1_std = np.sqrt(np.sum(np.square(np.concatenate([receive_b1_remote,receive_b1]) - np.sqrt(2)/2))/(receive_b1.size))
    return b0_std, b1_std

# def qpsk_llrs_from_constellation(constellation, *, clockwise=False, clip=False, clip_ts=2, belief_scale=1.0):
#     """
#     Build LLRs for QPSK mapping used at TX:
#         (b0,b1)->(±1 ± j)/sqrt(2)
#     b0 ↔ Imag sign, b1 ↔ Real sign.
#     Returns interleaved LLRs: [LLR_b0, LLR_b1, ...].
#     """
#     s = np.asarray(constellation, dtype=np.complex64)
#     radius = 0.5
#     if clockwise:
#         llr0 = s.real
#         llr1 = s.imag
#     else:
#         llr0 = s.imag
#         llr1 = s.real
#     avg0 = np.mean(np.abs(llr0))
#     avg1 = np.mean(np.abs(llr1))
#     std0 = np.std(np.abs(llr0))
#     std1 = np.std(np.abs(llr1))
#     # llr0 = np.where(np.abs(llr0) > 2 * avg0, 0, llr0)
#     # llr1 = np.where(np.abs(llr1) > 2 * avg1, 0, llr1)
#     # llr0 = np.where(np.abs(llr0) > avg0, np.sign(llr0) * avg0, llr0)
#     # llr1 = np.where(np.abs(llr1) > avg1, np.sign(llr1) * avg1, llr1)
#     # llr0 = llr0 /avg0
#     # llr1 = llr1 /avg1
#     llr0 = llr0 * np.sqrt(2)
#     llr1 = llr1 * np.sqrt(2)
#     # llr0 = llr0 * np.sqrt(2) / np.square(std0)
#     # llr1 = llr1 * np.sqrt(2) / np.square(std1)
#     if clip:
#         llr0 = np.clip(llr0, -(1 + clip_ts * std0) / np.square(std0), (1 + clip_ts * std0) / np.square(std0))
#         llr1 = np.clip(llr1, -(1 + clip_ts * std1) / np.square(std1), (1 + clip_ts * std1) / np.square(std1))
#     inter = np.empty(2 * s.size, dtype=np.float32)
#     inter[0::2] = llr0.real * belief_scale
#     inter[1::2] = llr1.real * belief_scale
#     return inter

# def qpsk_llrs_from_constellation(constellation, *, clockwise=False, clip=False, clip_ts=2, belief_scale=1.0):
#     """
#     Build LLRs for QPSK mapping used at TX:
#         (b0,b1)->(±1 ± j)/sqrt(2)
#     b0 ↔ Imag sign, b1 ↔ Real sign.
#     Returns interleaved LLRs: [LLR_b0, LLR_b1, ...].
#     """
#     s = np.asarray(constellation, dtype=np.complex64)
#     sigmab0, sigmab1 = estimate_constellation_std(constellation, clockwise=clockwise)
#     if clockwise:
#         llr0 = s.real
#         llr1 = s.imag
#     else:
#         llr0 = s.imag
#         llr1 = s.real
#     llr0 = llr0 * np.sqrt(2) / np.square(sigmab0)
#     llr1 = llr1 * np.sqrt(2) / np.square(sigmab1)
#     if clip:
#         llr0 = np.clip(llr0, -(1+clip_ts*sigmab0)/np.square(sigmab0), (1+clip_ts*sigmab0)/np.square(sigmab0))
#         llr1 = np.clip(llr1, -(1+clip_ts*sigmab1)/np.square(sigmab1), (1+clip_ts*sigmab1)/np.square(sigmab1))
#     inter = np.empty(2*s.size, dtype=np.float32)
#     inter[0::2] = llr0.real*belief_scale
#     inter[1::2] = llr1.real*belief_scale
#     return inter

def qpsk_llrs_from_constellation(constellation, *, clockwise=False,
                                 llr_clip=20.0, k_sigma_clip=4.0,
                                 belief_scale=1.0, use_mad=True):
    import numpy as np
    s = np.asarray(constellation, dtype=np.complex64)
    a, b = s.real, s.imag
    if clockwise:
        a, b = b, a
    A = 1/np.sqrt(2)

    def robust_sigma(x):
        centers = np.where(x >= 0.0, A, -A)
        r = x - centers
        if use_mad:
            med = np.median(r)
            mad = np.median(np.abs(r - med))
            sig = 1.4826 * mad
        else:
            lo, hi = np.percentile(r, [10, 90])
            sel = (r >= lo) & (r <= hi)
            sig = np.std(r[sel]) if np.any(sel) else np.std(r)
        return float(np.clip(sig, 1e-3, 1.0))

    sigma_r = robust_sigma(a)
    sigma_i = robust_sigma(b)

    a = np.clip(a, -A - k_sigma_clip*sigma_r, A + k_sigma_clip*sigma_r)
    b = np.clip(b, -A - k_sigma_clip*sigma_i, A + k_sigma_clip*sigma_i)

    llr1 = (np.sqrt(2) * a) / (sigma_r**2)   # Real -> b1
    llr0 = (np.sqrt(2) * b) / (sigma_i**2)   # Imag -> b0
    llr1 = np.clip(llr1, -llr_clip, llr_clip)
    llr0 = np.clip(llr0, -llr_clip, llr_clip)

    inter = np.empty(2 * s.size, dtype=np.float32)
    inter[0::2] = llr0 * belief_scale
    inter[1::2] = llr1 * belief_scale
    return inter, {'sigma_r': sigma_r, 'sigma_i': sigma_i}

def plot_correlation(corr):
    plt.plot(corr, color='blue', alpha=0.5, label='chirp_front')
    plt.show()

def plot_impulse_response(h_t, fs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    draw_in_TD(h_t.size / fs, h_t, title='Impulse response in time domain', ax=axes[0], x_label='time/s',
               y_label='h(t)')
    draw_in_FD(fs, h_t, title='Impulse response in freq domain', half=False, ax=axes[1], mode='Amplitude',
               y_label='H(f)/dB', x_label='Freq/Hz')
    plt.show()

def plot_unwrap_phase_fitting(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N):
    phase_shift = np.concatenate([phase_shift[N//2:],phase_shift[:N//2]])
    plt.title("Unwrap phase fitting line")
    x = np.linspace(-N//2,N//2,N,endpoint=False)
    plt.plot(x, np.abs(phase_shift), color='pink', label='amplitude', alpha=0.5)
    plt.plot(x, np.angle(phase_shift), color='orange', label='original', alpha=0.5)
    plt.plot(x, slope * x + intercept, linestyle='solid', label='fitting result', color='red')
    plt.plot(x, slope * x + intercept + np.pi, linestyle='dotted', color='red', alpha=0.5)
    plt.plot(x, slope * x + intercept - np.pi, linestyle='dotted', color='red', alpha=0.5)
    plt.scatter(x_auto, auto_unwrapped_phase, label='auto_unwrap', s=1, color='green', marker='*', alpha=0.5)
    plt.axhline(0, linestyle='dotted', color='black', linewidth=2)
    plt.axvline(0, linestyle='dotted', color='black', linewidth=2)
    plt.xlabel("sampling point")
    plt.ylabel("unwrapped phase")
    plt.legend()
    plt.show()

def plot_original_constellations(symbols, H_fs, pilot, effective_symbols_num, N, pic_idx, n_rows, n_cols):
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

def plot_corrected_constellations(symbols, origin_H_f, delta, fixed_phase_shift_factor, pilot, effective_symbols_num, N, pic_idx, n_rows, n_cols, symbol_len, num_symbols):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    for i, index in enumerate(pic_idx):
        corrected_H_f = correct_H_f(origin_H_f, delta, N, index, symbol_len)
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=non_approximate,
                                                    symbol_len=N)
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=corrected_constellation, emit_pilot=pilot[index, 1:N // 2], ax=ax,
                               title=f"constellation{index + 1}")
    fig.suptitle("corrected constellation")
    plt.tight_layout()
    plt.show()

def plot_received_signal(rx, ofdm_start, num_symbols, N, cp_len, delay):
    plt.plot(rx)
    plt.axvline(ofdm_start, linestyle='dotted', color='red')
    plt.axvline(ofdm_start + (num_symbols+64) * (N+cp_len)+delay, linestyle='dotted', color='red')
    plt.axvline(ofdm_start + num_symbols * (N + cp_len) + delay, linestyle='dotted', color='red')
    plt.show()

def plot_data_constellations(symbols, origin_H_f, delta, fixed_phase_shift_factor, emit_constellations, N, pic_idx, n_rows, n_cols, symbol_len, num_symbols):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))
    for i, index in enumerate(pic_idx):
        corrected_H_f = correct_H_f(origin_H_f, delta, N, index+num_symbols, symbol_len)
        constellation = get_constellation(symbols=symbols[index], H_f=corrected_H_f,
                                         approximation=non_approximate,
                                         symbol_len=N)
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=constellation, emit_pilot=emit_constellations[index], ax=ax,
                               title=f"constellation{index + 1}")
    fig.suptitle("data constellation")
    plt.tight_layout()
    plt.show()

def analysis_txt(plot=False, plot_opt=None):
    fs = 48000
    N = 4096
    cp_len = 1024
    num_symbols = 8
    symbol_len = N + cp_len
    fine_tune = 20

    rx = np.load(fr"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\LDPC\received_signal_chirp_l2_10_24k_fs48k_N4096_cp1024_S8diff_R1-2_Z27_802.11n_A_scrambler7b.npy")
    pilot = np.load(fr"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\LDPC\pilot_S8diff_txt_scrambler7b.npy")
    chirp_template = generate_chirp(fs, duration=2, f_l=10, f_h=24000)

    corr = correlate(rx, chirp_template, mode='full')
    if plot and plot_opt['correlation']:
        plot_correlation(corr)
    ofdm_start = np.argmax(corr) + 1
    chirp_start = ofdm_start - chirp_template.size

    print(f"ofdm start {ofdm_start}")
    print(f"corr peak at {chirp_start}")

    print(f"开始提取 OFDM 符号")
    # ofdm_start = ofdm_start - fine_tune
    ofdm_start = ofdm_start
    rx_pilot = rx[ofdm_start: ofdm_start + num_symbols * (N + cp_len)]

    effective_symbols_num = num_symbols

    # Demodulate and decode OFDM
    symbols = get_symbols(record=np.array(rx_pilot), cp_len=cp_len, N=N)
    H_fs = list()
    for i in range(effective_symbols_num):
        H_f = evaluate_H_f(known_symbols=symbols[i, :], pilot_signals=pilot[i])
        H_fs.append(H_f)

    H_fs = H_fs[0:effective_symbols_num]
    H_f_former = H_fs[: effective_symbols_num - 1]
    H_f_latter = H_fs[1:effective_symbols_num]
    phase_shift = np.mean(np.stack(H_f_latter, axis=0) / np.stack(H_f_former, axis=0), axis=0)

    x_auto, auto_unwrapped_phase,_ = phase_unwrap_auto(data=phase_shift)
    slope, intercept = fitting_line(x=x_auto, y=auto_unwrapped_phase, filter=True, residual_th=1.5)
    delta = slope / (symbol_len * (-2 * np.pi) / N)
    fixed_phase_shift_factor = intercept
    freq_bias = fs / (delta + 1) - fs
    print(f"delta:{delta}")
    print(f"fixed_phase_shift_factor:{fixed_phase_shift_factor}")
    print(f"fs of receiver - fs of emitter = {freq_bias}")
    print(f"delay {delta * effective_symbols_num * symbol_len} points "
          f"after {effective_symbols_num} symbols(N:{N} and cp{cp_len})")
    print()

    if plot and plot_opt['unwrap']:
        plot_unwrap_phase_fitting(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N)

    # Draw constellation distribution map
    n_rows = 2
    n_cols = 4
    pic_num = n_rows * n_cols
    pic_idx = np.linspace(start=0,
                          stop=0+effective_symbols_num//(pic_num-1)*(pic_num-1),
                          num=pic_num).astype(np.int32)
    origin_H_f = 0
    for index, H_f in enumerate(H_fs):
        # origin_H_f += H_f / np.exp(1j * (-2 * np.pi / N * delta * index * symbol_len * np.arange(N)
        #                                  + fixed_phase_shift_factor * index))
        origin_H_f += correct_H_f(H_f, delta, N, -index, symbol_len)
    origin_H_f /= len(H_fs)
    if plot and plot_opt['impulse_response']:
        plot_impulse_response(np.fft.ifft(origin_H_f), fs)
    if plot and plot_opt['raw_pilot_constellation']:
        plot_original_constellations(symbols, H_fs, pilot, effective_symbols_num, N, pic_idx, n_rows, n_cols)
    if plot and plot_opt['corrected_pilot_constellation']:
        plot_corrected_constellations(symbols, origin_H_f, delta, intercept, pilot, effective_symbols_num, N, pic_idx, n_rows, n_cols, symbol_len, num_symbols)

    b0stds, b1stds = caculate_constellation_std(symbols, origin_H_f, delta, fixed_phase_shift_factor, pilot, N, symbol_len, num_symbols)
    coeffsb0 = np.polyfit(np.arange(num_symbols), np.array(b0stds), deg=1)
    coeffsb1 = np.polyfit(np.arange(num_symbols), np.array(b1stds), deg=1)

    # Statistical BER for pilot symbols
    received_bits = list()
    emit_bits = QPSK_reflection(data=pilot[:effective_symbols_num, 1:N // 2]).flatten()
    for index in range(effective_symbols_num):
        raw_constellation = get_constellation(symbols=symbols[index, :], H_f=origin_H_f,
                                              approximation=simple_approximate,
                                              symbol_len=N)
        received_bits.append(QPSK_reflection(data=raw_constellation))
    received_bits = np.array(received_bits).flatten()
    BER = 1 - np.sum(np.equal(received_bits, emit_bits)) / received_bits.size
    print(f"bit error rate (uncorrected):{BER * 100:.4f}%")

    received_bits = list()
    for index in range(effective_symbols_num):
        corrected_H_f = correct_H_f(origin_H_f, delta, N, index, symbol_len)
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=simple_approximate,
                                                    symbol_len=N)
        received_bits.append(QPSK_reflection(data=corrected_constellation))
    received_bits = np.array(received_bits).flatten()
    BER = 1 - np.sum(np.equal(received_bits, emit_bits)) / received_bits.size
    print(f"bit error rate (corrected):{BER * 100:.4f}%")




    print("Start to extract data part")
    delay = (np.floor(delta * effective_symbols_num * symbol_len).astype(np.int32))
    if plot and plot_opt['received_signal']:
        plot_received_signal(rx, ofdm_start, num_symbols, N, cp_len, delay)

    # rx_data = rx[ofdm_start + num_symbols * (N + cp_len) + delay:]
    rx_data = rx[ofdm_start + num_symbols * (N + cp_len):]
    symbols = get_symbols(record=rx_data, N=N, cp_len=cp_len)
    received_bits = list()
    emit_bits = get_bits_from_file(r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\data\shakespace(short).txt")
    # Descramble the original bits for comparison
    emit_bits = scrambler(emit_bits, seed=0b1111111)
    emit_bits_ldpc, (K, Ncw) = ldpc_encode_bits(emit_bits)
    emit_constellations = QPSK_mapping(serial_to_parallel(emit_bits_ldpc, N=N))
    n_rows = 4
    n_cols = 4
    pic_num = n_rows * n_cols
    pic_idx = np.linspace(start=0,
                          stop=0 + emit_constellations.shape[0] // (pic_num - 1) * (pic_num - 1),
                          num=pic_num).astype(np.int32)
    if plot and plot_opt['data_constellation']:
        plot_data_constellations(symbols, origin_H_f, delta, fixed_phase_shift_factor, emit_constellations, N, pic_idx, n_rows, n_cols, symbol_len, num_symbols)

    # === LDPC SOFT-DECODING PATH (preserve offset correction) ===
    llr_blocks = []
    src_idx_blocks = []
    for index, symbol in enumerate(symbols):
        corrected_H_f = correct_H_f(origin_H_f, delta, N, index+num_symbols, symbol_len)
        constellation = get_constellation(symbols=symbol, H_f=corrected_H_f,
                                          approximation=normalize_approximate)

        llr_sym, st = qpsk_llrs_from_constellation(
            constellation, llr_clip=20.0, k_sigma_clip=4.0, belief_scale=1.0, use_mad=True
        )

        # 依据 EVM 估计做符号级降权（0.6~1.2 区间）
        evm2 = 0.5 * (st['sigma_r'] ** 2 + st['sigma_i'] ** 2)
        scale = 1.0 / (1.0 + 6.0 * evm2)  # 6.0 可调：噪声越大，scale 越小
        scale = np.clip(scale, 0.6, 1.2)
        llr_blocks.append(llr_sym * scale)
        # ... 得到 llr_sym 长度 = bits_per_symbol
        src_idx_blocks.append(np.full(llr_sym.size, index, dtype=np.int32))
    all_llrs = np.concatenate(llr_blocks) if llr_blocks else np.array([], dtype=np.float32)
    src_idx = np.concatenate(src_idx_blocks)


    # LDPC decode in blocks of Ncw
    c = ldpc.code(standard=LDPC_STANDARD, rate=LDPC_RATE, z=LDPC_Z, ptype=LDPC_PTYPE)
    assert Ncw == c.N and K == c.K, "make sure using same ldpc standard"
    nblocks = min(all_llrs.size // Ncw, emit_bits_ldpc.size // Ncw)
    if nblocks == 0:
        print("No data LLRs collected; aborting.")
        return
    llrs_used = all_llrs[:nblocks * Ncw].reshape(nblocks, Ncw)

    # 与 llrs_used 同步切块：
    src_used = src_idx[:nblocks * Ncw].reshape(nblocks, Ncw)

    # ground-truth 编码比特按 Ncw 重排（不覆盖原变量，避免后面复用出问题）
    gt_cw = emit_bits_ldpc.reshape(-1, Ncw)

    # 只打印前 100 个块（也要防止块数或 gt 数量不够）
    print_limit = min(100, nblocks, gt_cw.shape[0])


    # 可调：一次外部批量大小（和 dgl_microbatch 不同；这个是你喂给 decode 的批大小）
    BATCH = 512  # 结合显存自己调；也可更大，内部还有 c.dgl_microbatch 兜底
    c.dgl_microbatch = 256  # 避免一次性张量太大；也可以不设或设 None

    decoded_info = []
    c.dgl_device = 'cuda'  # 或 'cuda:0'
    c.dgl_llr_clip = 20.0
    c.dgl_max_iter = 200
    c.dgl_verbose = False  # True 时打印每若干轮的综合校验和
    c.dgl_log_every = 1  # 日志间隔（轮）
    c.dgl_check_every = 1  # syndrome 早停检查的频率（轮）
    for s in range(0, nblocks, BATCH):
        e = min(s + BATCH, nblocks)

        # 1) 组装 ch_batch: 形状 (b, N)
        # llrs_used 如果是 list[np.ndarray], 用 stack；如果本来就是 (nblocks, N) 的 ndarray，直接切片即可
        if isinstance(llrs_used, list):
            ch_batch = np.stack([llrs_used[i].copy() for i in range(s, e)], axis=0)
        else:
            ch_batch = llrs_used[s:e].copy()

        # 2) 预检查（按样本计算 pre-BER）
        hard_pre = (ch_batch < 0).astype(np.uint8)  # (b, N)
        pre_ber = np.mean(hard_pre != gt_cw[s:e], axis=1)  # (b,)

        # 3) 一次性批量解码（GPU）
        appB, itB = c.decode(ch_batch, dectype='sumprod2_dgl')  # appB: (b, N)
        xhatB = (appB < 0).astype(np.uint8)  # (b, N)
        post_ber = np.mean(xhatB != gt_cw[s:e], axis=1)  # (b,)

        # 4) 打印与收集
        for j, i in enumerate(range(s, e)):
            if i < print_limit:
                it_show = itB if np.isscalar(itB) else int(itB)  # GPU 批量路径返回的是统一的轮数
                print(f"[BLK {i:04d}] it={it_show:3d} preBER={pre_ber[j]:.4f} postBER={post_ber[j]:.4f}")
            decoded_info.append(xhatB[j, :K])


    received_bits = np.concatenate(decoded_info).astype(np.uint8)

    received_bits = received_bits[:emit_bits.size]
    # Descramble received bits
    emit_bits = scrambler(emit_bits, seed=0b1111111)
    received_bits = scrambler(received_bits, seed=0b1111111)
    print(symbols.shape[0])

    emit_bit0 = emit_bits.reshape(-1, 2)[:, 1]
    received_bit0 = received_bits.reshape(-1, 2)[:, 1]
    print(f"BER for bit 0: {1 - np.sum(np.equal(emit_bit0.flatten(), received_bit0.flatten())) / emit_bit0.size:.4f}")

    emit_bit1 = emit_bits.reshape(-1, 2)[:, 0]
    received_bit1 = received_bits.reshape(-1, 2)[:, 0]
    print(f"BER for bit 1: {1 - np.sum(np.equal(emit_bit1.flatten(), received_bit1.flatten())) / emit_bit1.size:.4f}")

    # Total BER
    BER = 1 - np.sum(np.equal(received_bits, emit_bits)) / received_bits.size
    print(f"Total BER: {BER * 100:.4f}%")

    # for i in range(0, 30):
    #     print(1 - np.sum(np.equal(received_bits[i * 4096:(i + 1) * 4096], emit_bits[4096 * i:(i + 1) * 4096])) / 4096)

    # Reconstruct bytes from descrambled bits
    bytes = np.packbits(received_bits)

    # bytes = get_bytes(np.array(received_bits).flatten())
    with open(output_dir + "/shakespeare1.txt", 'wb') as file:
        file.write(bytes.tobytes())

if __name__ == "__main__":
    assert os.path.exists(project_dir), "specify your proj dir"
    dirs = [output_dir, record_dir, data_dir]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    plot_opt ={
        'correlation':False,
        'impulse_response': False,
        'raw_pilot_constellation': False,
        'corrected_pilot_constellation': False,
        'data_constellation': True,
        'unwrap': True,
        'received_signal': False
    }

    analysis_txt(plot=True, plot_opt=plot_opt)