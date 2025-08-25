import sys
import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import correlate, chirp, lfilter
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

# below are relative import, ignore the warning, they won't influence the code
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import (get_symbols, get_constellation, simple_approximate,
                   QPSK_reflection, get_bytes, evaluate_H_f, non_approximate, normalize_approximate)
from utils import draw_in_TD, draw_in_FD, draw_constellation_map
from utils import phase_unwrap, fitting_line, normalize, phase_unwrap_auto
from utils import ldpc_encode_bits, scrambler
from utils import (generate_chirp, get_bits_from_file, serial_to_parallel,
                   QPSK_mapping, OFDM_modulate)
from utils import LDPC_PY_PATH

# ---------- LDPC library import and params (match TX) ----------
LDPC_PY_PATH = r'D:\\Pycharm\\SEU-CAM-25-Newton-s-Apple\\ldpc_jossy\\py'
if LDPC_PY_PATH and LDPC_PY_PATH not in sys.path:
    sys.path.append(LDPC_PY_PATH)

try:
    import ldpc  # from ldpc_jossy/py
except Exception as e:
    raise ImportError(
        f"无法导入 ldpc 包：{e}\\n"
        f"请检查 LDPC_PY_PATH 是否指向 ldpc_jossy/py 目录，并且包含 __init__.py。"
    )

LDPC_STANDARD = '802.11n'
LDPC_RATE = '1/2'
LDPC_Z = 27
LDPC_PTYPE = 'A'  # only for 802.16 rate 2/3 or 3/4

project_dir = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple"
output_dir = os.path.join(project_dir, "Channel Measurement/output/ldpc")
record_dir = os.path.join(project_dir, "Channel Measurement/record")
data_dir = os.path.join(project_dir, "Channel Measurement/data")

fs = 48000
N = 8192  # FFT size
POS_BINS = np.arange(1, N // 2)  # 正频 1..4095
DATA_START = 409
DATA_TAIL = 409  # TX 实际用于数据/导频的前缀子载波数
DATA_BINS = POS_BINS[DATA_START:-DATA_TAIL]  # 这些才是有效子载波
cp_len = 1024
num_symbols = 8
symbol_len = N + cp_len

# —— comb 导频参数（与发端一致）——
ITERATION = 10                # 每 5 个数据 OFDM 符号后插 1 个导频
COMB_PILOT_SEED_BASE = 128    # 发端 OFDM_modulate_data_with_comb 的 seed 起点

# --- Channel blending config ---
CHAN_BLEND_MODE = 'fixed'          # 'fixed'（固定权重）或 'distance'（按时间距离加权）
CHAN_BLEND_WEIGHTS = (0.4, 0.6)    # (from_start, from_nearest) 例如 40% + 60%
USE_DD_CPE = True                  # 是否开启判决导向 CPE 二次校正


def correct_H_f(origin_H_f, delta, N, index, symbol_len, fixed_phase_shift_factor=0.0):
    """
    fixed_phase_shift_factor: 每 OFDM 符号的常相位步进（phi_step），单位：弧度/符号
    """
    # 保持你原有的子载波索引顺序，避免与 H_f 的频率排列不一致
    k = np.linspace(-N//2, N//2, N, endpoint=False, dtype=np.int32)
    k = np.concatenate([k[N//2:], k[:N//2]])
    # 线性相位（与子载波索引 k 成正比）+ 每符号常相位
    phase_k = (-2 * np.pi / N) * delta * index * symbol_len * k
    phi = fixed_phase_shift_factor * index
    corrected_H_f = origin_H_f * np.exp(1j * (phase_k + phi))
    return corrected_H_f


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
    draw_in_FD(fs, h_t, title='Impulse response in freq domain', half=True, ax=axes[1], mode='Amplitude',
               y_label='H(f)/dB', x_label='Freq/Hz')
    plt.show()

def generate_pilot_symbol(N, seed=256):
    """生成与发端一致的对称随机导频（频域），保证 Hermitian 对称。"""
    rng = np.random.default_rng(seed)
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

def fit_drift_between(H_start, H_end, gap, *, N=N, symbol_len=None, return_phi=False):
    """
    由相隔 gap 个 OFDM 的两次信道估计，拟合得到：
      - delta：每符号的线性相位斜率（对应 SFO/CFO 残差）
      - phi_step：每符号公共相位步进（CPE）
    """
    if symbol_len is None:
        raise ValueError("symbol_len must be provided")
    phase_shift = H_end / H_start
    x_auto, auto_unwrapped_phase, _ = phase_unwrap_auto(data=phase_shift)
    slope, intercept = fitting_line(x=x_auto, y=auto_unwrapped_phase, filter=True, residual_th=1.5)

    # plot_unwrap_phase_fitting(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N)

    delta = slope / (symbol_len * (-2 * np.pi) / N) / gap
    phi_step = intercept / gap  # 每“一个”符号的常相位步进

    if return_phi:
        return float(delta), float(phi_step)
    return float(delta)


def plot_unwrap_phase_fitting(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N):
    phase_shift = np.concatenate([phase_shift[N//2:],phase_shift[:N//2]])
    plt.title("Unwrap phase fitting line")
    x = np.linspace(-N//2,N//2,N,endpoint=False)
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
                                              approximation=non_approximate)
        raw_constellation = raw_constellation[DATA_START:-DATA_TAIL]
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=raw_constellation, emit_pilot=pilot[index, DATA_BINS], ax=ax,
                               title=f"constellation{index + 1}")
    fig.suptitle("original constellation")
    plt.tight_layout()
    plt.show()

def plot_corrected_constellations(symbols, origin_H_f, delta, fixed_phase_shift_factor, pilot, effective_symbols_num, N, pic_idx, n_rows, n_cols, symbol_len, num_symbols):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    for i, index in enumerate(pic_idx):
        corrected_H_f = correct_H_f(origin_H_f, delta, N, index, symbol_len, fixed_phase_shift_factor)
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=non_approximate)
        corrected_constellation = corrected_constellation[DATA_START:-DATA_TAIL]
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=corrected_constellation, emit_pilot=pilot[index, DATA_BINS], ax=ax,
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
        corrected_H_f = correct_H_f(origin_H_f, delta, N, index+num_symbols, symbol_len, fixed_phase_shift_factor)
        constellation = get_constellation(symbols=symbols[index], H_f=corrected_H_f,
                                         approximation=non_approximate)
        constellation = constellation[DATA_START:-DATA_TAIL]
        ax = axes[i // n_cols, i % n_cols]
        emit_ref = \
        emit_constellations.ravel()[:emit_constellations.size // (DATA_BINS.size) * (DATA_BINS.size)].reshape(-1, (DATA_BINS.size))[index]
        draw_constellation_map(received=constellation, emit_pilot=emit_ref, ax=ax,
                               title=f"constellation{index + 1}")
    fig.suptitle("data constellation")
    plt.tight_layout()
    plt.show()

def plot_evm_vs_sub_carrier_idx(evm_avg:np.ndarray):
    evm_mean_total = np.mean(evm_avg)
    plt.figure(figsize=(10, 4))
    plt.scatter(DATA_BINS, 20 * np.log10(evm_avg), marker='o', color='royalblue', s=1, alpha=0.5)
    plt.axhline(20 * np.log10(evm_mean_total), linestyle='dotted', color='red')
    plt.xlabel("Subcarrier Index")
    plt.ylabel("EVM (dB)")
    plt.title("EVM vs Subcarrier Index")
    plt.grid(True)
    plt.show()

def _blend_weights(i, start_idx, near_idx, mode='fixed', weights=(0.4, 0.6)):
    if mode == 'fixed':
        w1, w2 = weights
    else:
        # 距离越近权重越大：w ∝ 1/(distance+eps)
        d1 = max(1, i - start_idx)     # 与段起点（通常是上一块导频或数据起点）的距离
        d2 = max(1, abs(i - near_idx)) # 与最近导频的距离
        inv1, inv2 = 1.0/d1, 1.0/d2
        s = inv1 + inv2
        w1, w2 = inv1/s, inv2/s
    return float(w1), float(w2)

def _estimate_cpe_dd(eq_constellation):
    """
    判决导向公共相位误差估计：用硬判做参考，然后对齐。
    仅对 QPSK 设计，其他调制可扩展。
    """
    s = np.asarray(eq_constellation).ravel()
    # QPSK 硬判星座（±1±j)/√2
    hard = (np.sign(s.real) + 1j*np.sign(s.imag)) / np.sqrt(2)
    # 估计接收与硬判的平均相位差
    # 取 sum(s * hard.conj()) 的相位，其负值作为需补偿的旋转角
    phi = -np.angle(np.vdot(hard, s))
    return float(phi)

def plot_pre_post_ber(pre_ber: np.ndarray, post_ber: np.ndarray):
    """
    画出 pre_ber 和 post_ber 的变化情况
    1. pre_ber 和 post_ber 单独的曲线图
    2. pre_ber 和 post_ber 叠加在同一张图
    """
    indices = np.arange(len(pre_ber))

    # --------- 分开画 ---------
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(indices, pre_ber, label="pre_ber", color="blue")
    axs[0].set_ylabel("pre_ber")
    axs[0].set_title("pre_ber vs index")
    axs[0].grid(True)

    axs[1].plot(indices, post_ber, label="post_ber", color="red")
    axs[1].set_xlabel("Index")
    axs[1].set_ylabel("post_ber")
    axs[1].set_title("post_ber vs index")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # --------- 叠加画 ---------
    plt.figure(figsize=(10, 4))
    plt.plot(indices, pre_ber, label="pre_ber", color="blue")
    plt.plot(indices, post_ber, label="post_ber", color="red")
    plt.xlabel("Index")
    plt.ylabel("BER")
    plt.title("pre_ber & post_ber vs index")
    plt.legend()
    plt.grid(True)
    plt.show()

def analysis_txt(plot=False, plot_opt=None):
    rx = np.load(
        fr"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\record\LDPC\received_png_chirp_l2_10_24k_fs48k_N8192_cp1024_S8diff_R1-2_Z27_802.11n_A_scrambler_middle_0.8_1.npy")
    pilot = np.load(
        fr"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\save\pilot\pilot_different_txt820_seed256_comb_1.npy")
    chirp_template = generate_chirp(fs, duration=2, f_l=10, f_h=24000)

    corr = correlate(rx, chirp_template, mode='full')
    if plot and plot_opt['correlation']:
        plot_correlation(corr)
    ofdm_start = np.argmax(corr) + 1
    chirp_start = ofdm_start - chirp_template.size

    print(f"ofdm start {ofdm_start}")
    print(f"corr peak at {chirp_start}")

    print(f"开始提取 OFDM 符号")
    # ofdm_start = ofdm_start - fine_tun
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

    x_auto, auto_unwrapped_phase, _ = phase_unwrap_auto(data=phase_shift)
    slope, intercept = fitting_line(x=x_auto, y=auto_unwrapped_phase, filter=True, residual_th=1.5)
    delta = slope / (symbol_len * (-2 * np.pi) / N)
    fixed_phase_shift_factor = intercept
    freq_bias = fs / (delta + 1) - fs
    print(f"delta:{delta}")
    print(f"fixed_phase_shift_factor:{fixed_phase_shift_factor}")
    print(f"fs of receiver - fs of emitter = {freq_bias}")
    print(f"delay {delta * effective_symbols_num * symbol_len} points "
          f"after {effective_symbols_num} symbols(N:{N} and cp{cp_len})")

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
        origin_H_f += correct_H_f(H_f, delta, N, -index, symbol_len, fixed_phase_shift_factor=-fixed_phase_shift_factor)
    origin_H_f /= len(H_fs)
    if plot and plot_opt['impulse_response']:
        plot_impulse_response(np.fft.ifft(origin_H_f), fs)
    if plot and plot_opt['raw_pilot_constellation']:
        plot_original_constellations(symbols, H_fs, pilot, effective_symbols_num, N, pic_idx, n_rows, n_cols)
    if plot and plot_opt['corrected_pilot_constellation']:
        plot_corrected_constellations(symbols, origin_H_f, delta, intercept, pilot, effective_symbols_num, N, pic_idx, n_rows, n_cols, symbol_len, num_symbols)

    # Statistical BER for pilot symbols
    # --- 未校正 BER ---
    received_bits = list()
    emit_bits = QPSK_reflection(data=pilot[:effective_symbols_num, DATA_BINS]).flatten()
    for index in range(effective_symbols_num):
        raw_constellation = get_constellation(symbols=symbols[index, :], H_f=origin_H_f,
                                              approximation=simple_approximate,
                                              symbol_len=N)
        raw_constellation = np.asarray(raw_constellation).ravel()[DATA_START:-DATA_TAIL]  # ★ 只取前3685
        received_bits.append(QPSK_reflection(data=raw_constellation))
    received_bits = np.array(received_bits).flatten()
    BER = 1 - np.sum(np.equal(received_bits, emit_bits)) / received_bits.size
    print(f"bit error rate (uncorrected):{BER * 100:.4f}%")

    # --- 校正后 BER ---
    received_bits = list()
    for index in range(effective_symbols_num):
        corrected_H_f = correct_H_f(origin_H_f, delta, N, index, symbol_len)
        corrected_constellation = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                                    approximation=simple_approximate,
                                                    symbol_len=N)
        corrected_constellation = np.asarray(corrected_constellation).ravel()[DATA_START:-DATA_TAIL]  # ★ 只取前3685
        received_bits.append(QPSK_reflection(data=corrected_constellation))
    received_bits = np.array(received_bits).flatten()
    BER = 1 - np.sum(np.equal(received_bits, emit_bits)) / received_bits.size
    print(f"bit error rate (corrected):{BER * 100:.4f}%")


    print("---------------------------------------------------------------")
    print("Start to extract data part")
    delay = (np.floor(delta * effective_symbols_num * symbol_len).astype(np.int32)) + 1
    if plot and plot_opt['received_signal']:
        plot_received_signal(rx, ofdm_start, num_symbols, N, cp_len, delay)

    rx_data = rx[ofdm_start + num_symbols * (N + cp_len) :]
    symbols_all = get_symbols(record=rx_data, N=N, cp_len=cp_len)

    M = 45
    # M = symbols_all.shape[0]
    pilot_pos = np.arange(ITERATION, M, ITERATION + 1, dtype=int)
    data_pos = np.setdiff1d(np.arange(M), pilot_pos)
    symbols_data = symbols_all[data_pos]
    symbols_comb_pilot = symbols_all[pilot_pos]

    # === LDPC SOFT-DECODING PATH (preserve offset correction) ===
    # === NEW: comb-aware data extraction & decoding ===
    llr_blocks = []
    src_idx_blocks = []
    sub_carr_freq_blocks = []
    Hf_comb = []
    evm_accum = np.zeros(DATA_BINS.size, dtype=float)
    count_accum = np.zeros(DATA_BINS.size, dtype=int)

    for j, comb_pilot in enumerate(symbols_comb_pilot):
        pilot_ref = generate_pilot_symbol(N, seed=COMB_PILOT_SEED_BASE + j)
        Hf_p = evaluate_H_f(known_symbols=comb_pilot, pilot_signals=pilot_ref)
        Hf_comb.append(Hf_p)
    Hf_comb = np.array(Hf_comb, dtype=complex) if len(Hf_comb)>0 else np.zeros((0,N), dtype=complex)

    # 以“前置 num_symbols 个导频”的最后一块为区间起点
    H_ref = H_fs[-1]
    prev_ref_idx = -1  # 参考是在数据段开始之前一拍

    seg_params = []  # (start_idx, end_idx, H_start, delta_j, phi_step_j)
    if len(Hf_comb) > 0:
        for j, pidx in enumerate(pilot_pos):
            gap = (pidx - prev_ref_idx)
            delta_j, phi_step_j = fit_drift_between(H_ref, Hf_comb[j], gap, N=N, symbol_len=symbol_len, return_phi=True)
            seg_params.append((prev_ref_idx, pidx, H_ref, delta_j, phi_step_j))
            prev_ref_idx = pidx
            H_ref = Hf_comb[j].copy()
        if prev_ref_idx < M - 1:
            seg_params.append((prev_ref_idx, M, H_ref, seg_params[-1][3], seg_params[-1][4]))
    else:
        seg_params.append((-1, M, H_fs[-1], float(delta), float(fixed_phase_shift_factor)))

    idx = np.array([(s + e) / 2 for s, e,_, d,_ in seg_params])
    delta_idx = np.array([d for s, e,_, d,_ in seg_params])

    for s, e,h, d,_ in seg_params:
        print(f"{s}, {e}, {d}")
        #plot_impulse_response(np.fft.ifft(h),fs)

    # 构造插值函数（线性为例，可改成 'quadratic', 'cubic'）
    f = interp1d(idx, delta_idx, kind='linear', fill_value='extrapolate')
    if plot and plot_opt['data_constellation']:
        n_rows = 4
        n_cols = 4
        pic_num = n_rows * n_cols
        pic_idx = np.linspace(start=0,
                              stop=0 + data_pos.size // (pic_num - 1) * (pic_num - 1),
                              num=pic_num).astype(np.int32)
        pic_idx = data_pos[pic_idx]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))

    emit_bits = get_bits_from_file(r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\data\shakespace(short).txt")
    emit_bits = scrambler(emit_bits, seed=0b1111111)
    emit_bits_ldpc, (K, Ncw) = ldpc_encode_bits(emit_bits)
    emit_constellations = QPSK_mapping(serial_to_parallel(emit_bits_ldpc, N=(DATA_BINS.size + 1) * 2))

    for (start_idx2, end_idx2_, H_start, delta_j, phi_step_j) in seg_params:
        end_inclusive = end_idx2_ - 1
        for i in range(start_idx2 + 1, min(end_inclusive, M - 1) + 1):

            # 1) 从“段起点参考”（H_start）外推到第 i 个数据符号
            dt = i - start_idx2
            H_from_start = correct_H_f(H_start, delta_j, N, dt, symbol_len, fixed_phase_shift_factor=phi_step_j)

            # 2) 从“最近 comb 导频参考”外推到第 i 个数据符号
            if len(pilot_pos) > 0:
                j_near = int(np.argmin(np.abs(pilot_pos - i)))
                idx_near = pilot_pos[j_near]
                H_near = Hf_comb[j_near]
                dt2 = i - idx_near
                H_from_near = correct_H_f(H_near, delta_j, N, dt2, symbol_len, fixed_phase_shift_factor=phi_step_j)
            else:
                H_from_near = H_from_start

            # 3) 融合（固定 40/60 或按距离加权）
            w1, w2 = _blend_weights(i, start_idx2, idx_near if len(pilot_pos) > 0 else start_idx2,
                                    mode=CHAN_BLEND_MODE, weights=CHAN_BLEND_WEIGHTS)
            H_used = (w1 * H_from_start + w2 * H_from_near) / (w1 + w2)

            # 4) 均衡 -> （可选）判决导向 CPE 二次修正
            constellation = get_constellation(
                symbols=symbols_all[i, :], H_f=H_used,
                approximation=non_approximate, symbol_len=N
            )
            constellation = np.asarray(constellation).ravel()[DATA_START:-DATA_TAIL]

            if USE_DD_CPE:
                phi_dd = _estimate_cpe_dd(constellation)
                constellation *= np.exp(1j * phi_dd)  # 调回去

            error = constellation - emit_constellations[np.where(data_pos == i)[0]]
            evm_sym = np.abs(error) ** 2 / np.abs(emit_constellations[np.where(data_pos == i)[0]]) ** 2
            evm_accum += evm_sym.flatten()
            count_accum += 1

            if plot and plot_opt['data_constellation'] and i in pic_idx.tolist():
                seq = np.where(pic_idx==i)[0]
                ax = axes[seq//n_cols, seq%n_cols][0]
                draw_constellation_map(received=constellation, emit_pilot=emit_constellations[np.where(data_pos==i)[0]],
                                       title=f'constellation{i}', ax=ax, limit_border=2)

            llr_sym, st = qpsk_llrs_from_constellation(
                constellation, llr_clip=20.0, k_sigma_clip=4.0, belief_scale=1.0, use_mad=True
            )
            # 你原有的 EVM-based scale 保留
            evm2 = 0.5 * (st['sigma_r'] ** 2 + st['sigma_i'] ** 2)
            scale = np.clip(1.0 / (1.0 + 6.0 * evm2), 0.6, 1.2)
            llr_blocks.append(llr_sym * scale)
            src_idx_blocks.append(np.full(llr_sym.size, i, dtype=np.int32))
            sub_carr_freq_blocks.append(np.repeat(np.linspace(0, fs, N)[DATA_BINS], 2))

    evm_avg = np.sqrt(evm_accum / count_accum)


    if plot and plot_opt['data_constellation']:
        fig.tight_layout()
        plt.show()
    if plot and plot_opt['evm_vs_sub_carr']:
        plot_evm_vs_sub_carrier_idx(evm_avg)

    all_llrs = np.concatenate(llr_blocks) if llr_blocks else np.array([], dtype=np.float32)
    src_idx = np.concatenate(src_idx_blocks) if src_idx_blocks else np.array([], dtype=np.int32)
    sub_carr_freq = np.concatenate(sub_carr_freq_blocks)

    # LDPC decode in blocks of Ncw
    c = ldpc.code(standard=LDPC_STANDARD, rate=LDPC_RATE, z=LDPC_Z, ptype=LDPC_PTYPE)
    assert Ncw == c.N and K == c.K, "make sure using same ldpc standard"
    nblocks = min(all_llrs.size // Ncw, emit_bits_ldpc.size // Ncw)
    if nblocks == 0:
        print("No data LLRs collected; aborting.")
        return
    llrs_used = all_llrs[:nblocks*Ncw].reshape(nblocks, Ncw)

    # 与 llrs_used 同步切块：
    src_used = src_idx[:nblocks * Ncw].reshape(nblocks, Ncw)
    sub_carr_freq_used = sub_carr_freq[:nblocks * Ncw].reshape(nblocks, Ncw)

    # ground-truth 编码比特按 Ncw 重排（不覆盖原变量，避免后面复用出问题）
    gt_cw = emit_bits_ldpc.reshape(-1, Ncw)

    # 只打印前 100 个块（也要防止块数或 gt 数量不够）
    print_limit = min(200, nblocks, gt_cw.shape[0])

    # 可选参数（实例属性，可在第一次调用前设置）
    c.dgl_device = 'cuda'  # 或 'cuda:0'
    c.dgl_llr_clip = 20.0
    c.dgl_max_iter = 60
    c.dgl_verbose = False  # True 时打印每若干轮的综合校验和
    c.dgl_log_every = 1  # 日志间隔（轮）
    c.dgl_check_every = 1  # syndrome 早停检查的频率（轮）

    BATCH = 512  # 结合显存自己调；也可更大，内部还有 c.dgl_microbatch 兜底
    c.dgl_microbatch = 256  # 避免一次性张量太大；也可以不设或设 None

    decoded_info = []
    bad_idxs = []

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

        # 5) 打印与收集
        for j, i in enumerate(range(s, e)):
            if i < print_limit:
                it_show = itB if np.isscalar(itB) else int(itB)  # GPU 批量路径返回的是统一的轮数
                print(f"[BLK {i:04d}] it={it_show:3d} preBER={pre_ber[j]:.4f} postBER={post_ber[j]:.4f}")
            decoded_info.append(xhatB[j, :K])
            if post_ber[j]:
                bad_idxs.append(i)

    if plot and plot_opt['BER_show']:
        plot_pre_post_ber(pre_ber, post_ber)

    for bi in bad_idxs:
        uniq, cnt = np.unique(src_used[bi], return_counts=True)
        top = uniq[np.argsort(-cnt)[:3]]
        if len(top) == 2:
            freq1 = np.min(sub_carr_freq_used[bi][np.where(src_used[bi].flatten() == uniq[0])[0]])
            freq2 = np.max(sub_carr_freq_used[bi][np.where(src_used[bi].flatten() == uniq[1])[0]])
            print(f"[BLK {bi}] dominated by OFDM symbols: {top}, counts={cnt[np.argsort(-cnt)[:3]]}, "
                  f"corresponding sub carrier freq: {freq1:.2f} - {fs / 2:.2f} (first OFDM symbol) and {fs / N:.2f} - {freq2} (second OFDM symbol)")
        elif len(top) == 1:
            min_freq = np.min(sub_carr_freq_used[bi])
            max_freq = np.max(sub_carr_freq_used[bi])
            print(f"[BLK {bi}] dominated by OFDM symbols: {top}, counts={cnt[np.argsort(-cnt)[:3]]}, "
                  f"corresponding sub carrier freq: {min_freq:.2f} - {max_freq:.2f}")
        else:
            raise ValueError(f"data split in {len(top)} OFDM symbols, not supported now")

    received_bits = np.concatenate(decoded_info).astype(np.uint8)

    received_bits = received_bits[:emit_bits.size]
    # Descramble received bits
    emit_bits = scrambler(emit_bits, seed=0b1111111)
    received_bits = scrambler(received_bits, seed=0b1111111)
    print(symbols.shape[0])

    emit_bit0 = emit_bits.reshape(-1,2)[:,1]
    received_bit0 = received_bits.reshape(-1,2)[:,1]
    print(f"BER for bit 0: {1 - np.sum(np.equal(emit_bit0.flatten(), received_bit0.flatten())) / emit_bit0.size:.4f}")

    emit_bit1 = emit_bits.reshape(-1,2)[:,0]
    received_bit1 = received_bits.reshape(-1,2)[:,0]
    print(f"BER for bit 1: {1 - np.sum(np.equal(emit_bit1.flatten(), received_bit1.flatten())) / emit_bit1.size:.4f}")

    # Total BER
    BER = 1 - np.sum(np.equal(received_bits, emit_bits)) / received_bits.size
    print(f"Total BER: {BER * 100:.4f}%")

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
        'received_signal': True,
        'evm_vs_sub_carr': True,
        'BER_show': True,
    }

    analysis_txt(plot=True , plot_opt=plot_opt)