import sys
import os
import math
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
from utils import ldpc_encode_bits, scrambler_random,scrambler
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
        f"无法导入 ldpc 包：{e}\n"
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
TXT_INPUT_PATH = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\data\shakespace_poem_middle.txt"
TIFF_INPUT_PATH = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\data\Jossy origin.tiff"

# === 与 Tx 对齐的重要常量（新增/核对） ===
NUM_INIT_PILOTS = 8          # 起始 OFDM 导频个数
HEADER_BITS     = 64         # 64-bit 文件头（MSB-first）
SCRAMBLE_SEED   = 256        # scrambler_random 的种子
COMB_ITERATION  = 10         # comb pilot 周期
COMB_PILOT_SEED_BASE = 128   # comb pilot 的 seed 起点（与Tx一致）

fs = 48000
N = 8192  # FFT size
POS_BINS = np.arange(1, N // 2)  # 正频 1..4095
DATA_START = 409
DATA_TAIL = 409  # TX 实际用于数据/导频的前缀子载波数
DATA_BINS = POS_BINS[DATA_START:-DATA_TAIL]  # 这些才是有效子载波
cp_len = 1024
num_symbols = NUM_INIT_PILOTS
symbol_len = N + cp_len

# --- Channel blending config ---
CHAN_BLEND_MODE = 'distance'          # 'fixed'（固定权重）或 'distance'（按时间距离加权）
CHAN_BLEND_WEIGHTS = (0.4, 0.6)       # (from_start, from_nearest)
USE_DD_CPE = True                     # 判决导向 CPE（后面我们用PLL替代，保持兼容开关）

def correct_H_f(origin_H_f, delta, N, index, symbol_len, fixed_phase_shift_factor=0.0):
    k = np.linspace(-N//2, N//2, N, endpoint=False, dtype=np.int32)
    k = np.concatenate([k[N//2:], k[:N//2]])
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
        return float(np.clip(sig, 2e-3, 1.0))

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
    if symbol_len is None:
        raise ValueError("symbol_len must be provided")
    phase_shift = H_end / H_start
    x_auto, auto_unwrapped_phase, _ = phase_unwrap_auto(data=phase_shift )
    slope, intercept = fitting_line(x=x_auto, y=auto_unwrapped_phase, filter=True, residual_th=1.5)

    # plot_unwrap_phase_fitting_zoom(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N,window=100)
    # plot_unwrap_phase_fitting(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N)

    delta = slope / (symbol_len * (-2 * np.pi) / N) / gap
    phi_step = intercept / gap
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

def plot_unwrap_phase_fitting_zoom(phase_shift, slope, intercept,
                                   x_auto, auto_unwrapped_phase, N, window=20):
    """
    只画出 0 点附近的解缠相位拟合图
    window: 显示范围（采样点数），例如 20 表示 [-20, 20]
    """
    phase_shift = np.concatenate([phase_shift[N // 2:], phase_shift[:N // 2]])
    plt.title("Unwrap phase fitting (zoom around 0)")
    x = np.linspace(-N // 2, N // 2, N, endpoint=False)

    # 画线
    plt.plot(x, np.angle(phase_shift), color='orange', label='original', alpha=0.5)
    plt.plot(x, slope * x + intercept, linestyle='solid', label='fitting result', color='red')
    plt.plot(x, slope * x + intercept + np.pi, linestyle='dotted', color='red', alpha=0.5)
    plt.plot(x, slope * x + intercept - np.pi, linestyle='dotted', color='red', alpha=0.5)
    plt.scatter(x_auto, auto_unwrapped_phase, label='auto_unwrap', s=1,
                color='green', marker='*', alpha=0.5)

    # 画辅助线
    plt.axhline(0, linestyle='dotted', color='black', linewidth=2)
    plt.axvline(0, linestyle='dotted', color='black', linewidth=2)

    plt.xlabel("sampling point")
    plt.ylabel("unwrapped phase")
    plt.legend()

    # 只显示 0 附近
    plt.xlim(-window, window)
    plt.show()


def plot_original_constellations(symbols, H_fs, pilot, effective_symbols_num, N, pic_idx, n_rows, n_cols):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    for i, index in enumerate(pic_idx):
        raw_constellation = get_constellation(symbols=symbols[index, :], H_f=H_fs[0],
                                              approximation=non_approximate)
        raw_constellation = raw_constellation[DATA_START:-DATA_TAIL]
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=raw_constellation, emit_pilot=pilot[index, DATA_BINS], ax=ax,
                               title=f"constellation{index + 1}",limit_border=2)
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
                               title=f"constellation{index + 1}",limit_border=2)
    fig.suptitle("corrected constellation")
    plt.tight_layout()
    plt.show()

def plot_received_signal(rx, ofdm_start, num_symbols, N, cp_len, delay):
    plt.plot(rx)
    plt.axvline(ofdm_start, linestyle='dotted', color='red')
    plt.axvline(ofdm_start + (num_symbols+64) * (N+cp_len)+delay, linestyle='dotted', color='red')
    plt.axvline(ofdm_start + num_symbols * (N + cp_len) + delay, linestyle='dotted', color='red')
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
        d1 = max(1, i - start_idx)
        d2 = max(1, abs(i - near_idx))
        inv1, inv2 = 1.0/d1, 1.0/d2
        s = inv1 + inv2
        w1, w2 = inv1/s, inv2/s
    return float(w1), float(w2)

# === metrics helpers ===
def evm_and_snr(received: np.ndarray, reference: np.ndarray):
    ref = np.asarray(reference).ravel()
    rx  = np.asarray(received).ravel()
    err = rx - ref
    evm_rms = np.abs(err) / (np.abs(ref) + 1e-12)
    evm_rms = np.maximum(evm_rms, 1e-9)
    snr_lin = 1.0 / (evm_rms ** 2)
    snr_db  = 10.0 * np.log10(snr_lin)
    return evm_rms, snr_lin, snr_db

def pilot_quality_from_constellation(eq_const, ref):
    evm_rms, _, snr_db = evm_and_snr(eq_const, ref)
    snr_med_db = float(np.median(snr_db))
    q = 10.0 ** (snr_med_db / 20.0)
    return q, snr_med_db

def blend_weights_quality(i, start_idx, near_idx, q_start=1.0, q_near=1.0):
    d1 = max(1, i - start_idx)
    d2 = max(1, abs(i - near_idx))
    s1 = (q_start) / d1
    s2 = (q_near)  / d2
    s  = s1 + s2
    return float(s1/s), float(s2/s)

def apply_mmse_shrinkage(eq_constellation, H_used_data_bins, sigma_r, sigma_i):
    sigma2 = 0.5 * (sigma_r**2 + sigma_i**2)
    N0 = 2.0 * sigma2 + 1e-12
    G = (np.abs(H_used_data_bins)**2) / (np.abs(H_used_data_bins)**2 + N0)
    return eq_constellation * G

# === NEW: ground-truth helpers ===
def _u64_to_bits_msb(n):
    b = np.zeros(64, dtype=np.uint8)
    for k in range(64):
        b[k] = (n >> (63 - k)) & 1
    return b

def _build_gt(tx_bits_path, data_bins_size, *, scramble_seed=256):
    """
    构造与 TX 完全一致的 data 参考（含 64-bit 文件头）：
      [64-bit 头 | payload] --(scrambler_random)-->  扰码
                        └──> LDPC 编码 → 串并 → QPSK → emit_constellations
    返回:
      info_bits_full   : 头+payload（扰码前）
      emit_bits_ldpc   : 编码后的比特流
      emit_const       : 参考星座矩阵，shape=[num_data_tx, data_bins_size]
      num_data_tx      : 实际 data OFDM 个数（包含头部对应的码块）
    """
    payload_bits = get_bits_from_file(tx_bits_path).astype(np.uint8)
    header_bits  = _u64_to_bits_msb(int(payload_bits.size))
    info_bits_full = np.concatenate([header_bits, payload_bits])           # 头在前
    info_bits_scr  = scrambler_random(info_bits_full, seed=scramble_seed)  # 与 TX 一致：先拼，再扰

    emit_bits_ldpc, (K, Ncw) = ldpc_encode_bits(info_bits_scr)

    # 与 TX 的并行长度一致：(data_bins + 1) * 2；只取前 data_bins*2 位映射到有效子载波
    per_ofdm_bits  = (data_bins_size + 1) * 2
    mat_bits       = serial_to_parallel(emit_bits_ldpc, N=per_ofdm_bits)          # [num_data_tx, per_ofdm_bits]
    mat_effective  = mat_bits[:, :data_bins_size * 2]                             # 丢掉那“+1”对应的2比特
    emit_const     = QPSK_mapping(mat_effective)                                  # [num_data_tx, data_bins_size]
    num_data_tx    = emit_const.shape[0]
    return info_bits_full, emit_bits_ldpc, emit_const, num_data_tx

def _make_data_index_map(data_pos, num_data_tx):
    """
    只保留前 num_data_tx 个数据符号，返回:
      data_pos_trunc: 截断后的 data_pos (长度 = num_data_tx)
      data_index_map: {全局OFDM索引 -> 行号(0..num_data_tx-1)}
    """
    data_pos_trunc = data_pos[:num_data_tx]
    data_index_map = {int(sym_idx): int(row) for row, sym_idx in enumerate(data_pos_trunc)}
    return data_pos_trunc, data_index_map

class DD_CPE_PLL:
    def __init__(self, alpha=0.15, snr_th_db=6.0):
        self.theta = 0.0
        self.alpha = float(alpha)
        self.snr_th_db = float(snr_th_db)

    def step(self, const, snr_med_db, hard=None):
        s = np.asarray(const).ravel()
        if hard is None:
            hard = (np.sign(s.real) + 1j*np.sign(s.imag)) / np.sqrt(2)
        e = np.angle(np.vdot(hard, s * np.exp(-1j * self.theta)))
        if (snr_med_db >= self.snr_th_db) and (abs(e) < np.pi/2):
            self.theta += self.alpha * e
        self.theta = (self.theta + np.pi) % (2*np.pi) - np.pi
        return s * np.exp(-1j * self.theta)

def llr_scale_from_snr(snr_db_per_sc, lo=2.0, hi=10.0, min_scale=0.3, max_scale=1.0):
    s = (np.clip(snr_db_per_sc, lo, hi) - lo) / (hi - lo + 1e-9)
    return min_scale + (max_scale - min_scale) * s

def esno_from_sigmas(sig_r: float, sig_i: float):
    sigma2 = 0.5 * (sig_r**2 + sig_i**2)
    esno_lin = 1.0 / (2.0 * sigma2 + 1e-12)
    esno_db  = 10.0 * np.log10(esno_lin)
    return esno_lin, esno_db

def plot_snr_over_time(snr_db_per_symbol: np.ndarray, title="SNR over OFDM symbols"):
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(snr_db_per_symbol)), snr_db_per_symbol, marker='o', linewidth=1.5)
    plt.xlabel("OFDM Symbol Index")
    plt.ylabel("SNR (dB)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_snr_over_subcarrier(snr_db_per_sc: np.ndarray, title="SNR over subcarriers"):
    plt.figure(figsize=(10, 4))
    plt.plot(DATA_BINS, snr_db_per_sc, linewidth=1.0)
    plt.xlabel("Subcarrier Index")
    plt.ylabel("SNR (dB)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_pre_post_ber(pre_ber: np.ndarray, post_ber: np.ndarray):
    indices = np.arange(len(pre_ber))
    fig, axs = plt.subplots(2, 1, figsize=(17, 10), sharex=True)
    axs[0].plot(indices, pre_ber, label="pre_ber", color="blue"); axs[0].set_ylabel("pre_ber"); axs[0].set_title("pre_ber vs index"); axs[0].grid(True)
    axs[1].plot(indices, post_ber, label="post_ber", color="red"); axs[1].set_xlabel("Index"); axs[1].set_ylabel("post_ber"); axs[1].set_title("post_ber vs index"); axs[1].grid(True)
    plt.tight_layout(); plt.show()
    plt.figure(figsize=(15, 6))
    plt.plot(indices, pre_ber, label="pre_ber", color="blue")
    plt.plot(indices, post_ber, label="post_ber", color="red")
    plt.xlabel("Index"); plt.ylabel("BER"); plt.title("pre_ber & post_ber vs index"); plt.legend(); plt.grid(True); plt.show()

# === 解析 64-bit 头（MSB-first） ===
def u64_from_bits_msb(bits64):
    b = ''.join('1' if int(x) else '0' for x in np.asarray(bits64, dtype=np.uint8).tolist())
    return int(b, 2)

def maybe_gt_cw(emit_bits_ldpc, start_blocks, end_blocks, Ncw, ground_truth: bool):
    """仅在 ground_truth=True 且 emit_bits_ldpc 可用时返回 GT 码字矩阵，否则返回 None。"""
    if (not ground_truth) or (emit_bits_ldpc is None):
        return None
    nblocks = max(0, int(end_blocks - start_blocks))
    if nblocks == 0:
        return None
    start = int(start_blocks * Ncw)
    end   = int(end_blocks   * Ncw)
    return build_gt_codewords(emit_bits_ldpc[start:end], nblocks, Ncw)

def build_gt_codewords(emit_bits_ldpc: np.ndarray, nblocks: int, Ncw: int) -> np.ndarray:
    """
    把编码后的比特流切成 (nblocks, Ncw) 的“真值码字矩阵”。
    若长度不足（理论上不该发生），则右侧用 0 填充。
    """
    total = nblocks * Ncw
    stream = np.asarray(emit_bits_ldpc, dtype=np.uint8).ravel()[:total]
    if stream.size < total:
        pad = np.zeros(total - stream.size, dtype=np.uint8)
        stream = np.concatenate([stream, pad], axis=0)
    return stream.reshape(nblocks, Ncw)

def analysis_txt(plot=False, plot_opt=None, ground_truth=True, tx_bits_path=TXT_INPUT_PATH):
    rx = np.load(
        fr"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\record\LDPC\received_txt_chirp_l2_10_24k_fs48k_N8192_cp1024_S8diff_R1-2_Z27_802.11n_A_random_middle_0.8_long_head_2.npy")
    pilot = np.load(
        fr"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\save\pilot\pilot_different_N8192_fixed.npy")
    chirp_template = generate_chirp(fs, duration=2, f_l=10, f_h=24000)

    corr = correlate(rx, chirp_template, mode='full')
    if plot and plot_opt['correlation']:
        plot_correlation(corr)
    ofdm_start = np.argmax(corr) + 1
    chirp_start = ofdm_start - chirp_template.size

    print(f"ofdm start {ofdm_start}")
    print(f"corr peak at {chirp_start}")

    print(f"开始提取 OFDM 符号")
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

    n_rows = 2; n_cols = 4; pic_num = n_rows * n_cols
    pic_idx = np.linspace(start=0, stop=0+effective_symbols_num//(pic_num-1)*(pic_num-1), num=pic_num).astype(np.int32)
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

    # Pilot SNR (略) —— 原逻辑保留
    snr_pilot_per_sym = []
    for index in range(effective_symbols_num):
        corrected_H_f = correct_H_f(origin_H_f, delta, N, index, symbol_len, fixed_phase_shift_factor)
        const = get_constellation(symbols=symbols[index, :], H_f=corrected_H_f,
                                  approximation=non_approximate, symbol_len=N)
        const = np.asarray(const).ravel()[DATA_START:-DATA_TAIL]
        ref   = pilot[index, DATA_BINS]
        evm_rms, _, snr_db = evm_and_snr(const, ref)
        snr_pilot_per_sym.append(float(np.median(snr_db)))
        _, st_loc = qpsk_llrs_from_constellation(const, use_mad=True)
        _, esno_db_loc = esno_from_sigmas(st_loc['sigma_r'], st_loc['sigma_i'])
        print(f"[PILOT {index:02d}] SNR_med={np.median(snr_db):.2f} dB | Es/N0≈{esno_db_loc:.2f} dB")
    snr_pilot_per_sym = np.array(snr_pilot_per_sym, dtype=float)
    print(f"[PILOT] SNR over symbols: mean={snr_pilot_per_sym.mean():.2f} dB | "
          f"median={np.median(snr_pilot_per_sym):.2f} dB | min={snr_pilot_per_sym.min():.2f} dB")
    if plot and plot_opt.get('snr_time_pilot', False):
        plot_snr_over_time(snr_pilot_per_sym, title="Pilot SNR over OFDM symbols")

    print("---------------------------------------------------------------")
    print("Start to extract data part")
    delay = (np.floor(delta * effective_symbols_num * symbol_len).astype(np.int32)) + 1
    if plot and plot_opt['received_signal']:
        plot_received_signal(rx, ofdm_start, num_symbols, N, cp_len, delay)

    rx_data = rx[ofdm_start + num_symbols * (N + cp_len) :]
    symbols_all = get_symbols(record=rx_data, N=N, cp_len=cp_len)

    # === M 自动 ===
    M = symbols_all.shape[0]

    pilot_pos = np.arange(COMB_ITERATION, M, COMB_ITERATION + 1, dtype=int)
    data_pos = np.setdiff1d(np.arange(M), pilot_pos)
    symbols_data = symbols_all[data_pos]
    symbols_comb_pilot = symbols_all[pilot_pos]

    # === comb pilot 估计 ===
    llr_blocks = []
    src_idx_blocks = []
    sub_carr_freq_blocks = []
    Hf_comb = []
    evm_accum = np.zeros(DATA_BINS.size, dtype=float)
    count_accum = np.zeros(DATA_BINS.size, dtype=int)

    snr_data_per_symbol = []
    snr_sc_sum = np.zeros(DATA_BINS.size)
    snr_sc_cnt = np.zeros(DATA_BINS.size, dtype=int)

    for j, comb_pilot in enumerate(symbols_comb_pilot):
        pilot_ref = generate_pilot_symbol(N, seed=COMB_PILOT_SEED_BASE + j)
        Hf_p = evaluate_H_f(known_symbols=comb_pilot, pilot_signals=pilot_ref)
        Hf_comb.append(Hf_p)
    Hf_comb = np.array(Hf_comb, dtype=complex) if len(Hf_comb)>0 else np.zeros((0,N), dtype=complex)

    pilot_qualities = []; pilot_snrdb = []
    if len(Hf_comb) > 0:
        for j, comb_pilot in enumerate(symbols_comb_pilot):
            const_p = get_constellation(symbols=comb_pilot, H_f=Hf_comb[j],
                                        approximation=non_approximate, symbol_len=N)
            const_p = np.asarray(const_p).ravel()[DATA_START:-DATA_TAIL]
            ref_p   = generate_pilot_symbol(N, seed=COMB_PILOT_SEED_BASE + j)[DATA_BINS]
            q, snr_db_med = pilot_quality_from_constellation(const_p, ref_p)
            pilot_qualities.append(q); pilot_snrdb.append(snr_db_med)
        pilot_qualities = np.array(pilot_qualities, dtype=float)
        pilot_snrdb     = np.array(pilot_snrdb, dtype=float)

    # 以“前置 num_symbols 个导频”的最后一块为区间起点
    H_ref = H_fs[-1]
    prev_ref_idx = -1
    seg_params = []
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

    for s, e,h, d,_ in seg_params:
        print(f"data_pilot_number:{(s+1)//COMB_ITERATION},start:{s},end: {e},delat: {d}")

    # === GT 构建（仅在 ground_truth=True 时） ===
    emit_bits = emit_bits_ldpc = emit_constellations = None
    num_data_tx = None
    if ground_truth:
        emit_bits, emit_bits_ldpc, emit_constellations, num_data_tx = _build_gt(
            tx_bits_path, data_bins_size=DATA_BINS.size, scramble_seed=SCRAMBLE_SEED
        )
        # 用 num_data_tx 截断 data_pos，建立映射，防止越界
        data_pos_trunc, data_index_map = _make_data_index_map(data_pos, num_data_tx)
    else:
        data_pos_trunc, data_index_map = data_pos, {}  # 用不到映射

    # === 初始化 CPE PLL ===
    cpe_pll = DD_CPE_PLL(alpha=0.15, snr_th_db=6.0)

    if plot and plot_opt['data_constellation']:
        n_rows = 4
        n_cols = 4
        pic_num = n_rows * n_cols
        pic_idx = np.linspace(start=0,
                              stop=0 + data_pos_trunc.size // (pic_num - 1) * (pic_num - 1),
                              num=pic_num).astype(np.int32)
        pic_idx = data_pos_trunc[pic_idx]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))

    # === 主循环：构造所有 LLR（不改你的主体流程） ===
    for (start_idx2, end_idx2_, H_start, delta_j, phi_step_j) in seg_params:
        end_inclusive = end_idx2_ - 1
        for i in range(start_idx2 + 1, min(end_inclusive, M - 1) + 1):
            if i not in data_pos:
                continue
            if i in processed_set:  # ★ 避免重复处理 Step1 用过的符号
                continue
            if ground_truth and (i not in data_index_map):
                continue

            # 1) 从段起点外推
            dt = i - start_idx2
            H_from_start = correct_H_f(H_start, delta_j, N, dt, symbol_len, fixed_phase_shift_factor=phi_step_j)

            # 2) 从最近 comb 导频外推
            if len(pilot_pos) > 0:
                j_near = int(np.argmin(np.abs(pilot_pos - i)))
                idx_near = pilot_pos[j_near]
                H_near = Hf_comb[j_near]
                dt2 = i - idx_near
                H_from_near = correct_H_f(H_near, delta_j, N, dt2, symbol_len, fixed_phase_shift_factor=phi_step_j)
            else:
                H_from_near = H_from_start

            # 3) 融合
            if len(pilot_pos) > 0:
                q_start = pilot_qualities[j_near - 1] if (j_near - 1) >= 0 else pilot_qualities[0]
                q_near = pilot_qualities[j_near]
                w1, w2 = blend_weights_quality(i, start_idx2, idx_near, q_start=q_start, q_near=q_near)
            else:
                w1, w2 = _blend_weights(i, start_idx2, start_idx2, mode=CHAN_BLEND_MODE, weights=CHAN_BLEND_WEIGHTS)

            H_used = (w1 * H_from_start + w2 * H_from_near) / (w1 + w2)

            # 4) 均衡
            constellation = get_constellation(symbols=symbols_all[i, :], H_f=H_used,
                                              approximation=non_approximate, symbol_len=N)
            constellation = np.asarray(constellation).ravel()[DATA_START:-DATA_TAIL]

            # --- 用发端参考（文件头+payload都在 emit_constellations 里） ---
            if ground_truth:
                ref_sc = emit_constellations[data_index_map[i]].ravel()
                snr_med_for_pll = float(np.median(evm_and_snr(constellation, ref_sc)[2]))
            else:
                hard = (np.sign(constellation.real) + 1j * np.sign(constellation.imag)) / np.sqrt(2)
                snr_med_for_pll = float(np.median(evm_and_snr(constellation, hard)[2]))

            # PLL
            constellation = cpe_pll.step(constellation, snr_med_db=snr_med_for_pll)

            # MMSE
            _, st_tmp = qpsk_llrs_from_constellation(constellation, use_mad=True)
            H_used_bins = H_used[DATA_BINS]
            constellation = apply_mmse_shrinkage(constellation, H_used_bins, st_tmp['sigma_r'], st_tmp['sigma_i'])

            if ground_truth:
                error = constellation - emit_constellations[data_index_map[i]]
                evm_sym = np.abs(error) ** 2 / (np.abs(emit_constellations[data_index_map[i]]) ** 2 + 1e-12)
                evm_accum += evm_sym.flatten()
                count_accum += 1

                evm_rms_sym, _, snr_db_sym = evm_and_snr(constellation, ref_sc)
                snr_data_per_symbol.append(float(np.median(snr_db_sym)))
                snr_sc_sum += snr_db_sym
                snr_sc_cnt += 1
                w_sc = llr_scale_from_snr(snr_db_sym, lo=2.0, hi=10.0, min_scale=0.4, max_scale=1.0)
            else:
                w_sc = np.ones(DATA_BINS.size, dtype=float)

            # 如果要画图
            if plot and plot_opt['data_constellation'] and ground_truth and (i in data_index_map) and (
                    i in pic_idx.tolist()):
                seq = np.where(pic_idx == i)[0]
                ax = axes[seq // n_cols, seq % n_cols][0]
                draw_constellation_map(received=constellation,
                                       emit_pilot=emit_constellations[data_index_map[i]],
                                       title=f'constellation{i}', ax=ax, limit_border=2)

            # 生成 LLR（保持原逻辑）
            llr_sym, st = qpsk_llrs_from_constellation(
                constellation, llr_clip=20.0, k_sigma_clip=4.0, belief_scale=0.5, use_mad=True
            )
            ...

            # --- 5) LLR 仍使用“MMSE后”的 constellation（你的原逻辑不变） ---
            llr_sym, st = qpsk_llrs_from_constellation(
                constellation, llr_clip=20.0, k_sigma_clip=4.0, belief_scale=0.5, use_mad=True
            )

            evm2 = 0.5 * (st['sigma_r'] ** 2 + st['sigma_i'] ** 2)
            scale = np.clip(1.0 / (1.0 + 6.0 * evm2), 0.6, 1.2)

            llr_sym_2col = llr_sym.reshape(-1, 2)
            llr_sym_2col *= w_sc[:, None]
            llr_sym = llr_sym_2col.ravel()

            llr_blocks.append(llr_sym * scale)
            src_idx_blocks.append(np.full(llr_sym.size, i, dtype=np.int32))
            sub_carr_freq_blocks.append(np.repeat(np.linspace(0, fs, N)[DATA_BINS], 2))

    evm_avg = np.sqrt(evm_accum / np.maximum(count_accum, 1))
    if plot and plot_opt.get('snr_time_data', False) and ground_truth:
        snr_data_per_symbol = np.array(snr_data_per_symbol, dtype=float) if len(snr_data_per_symbol)>0 else np.array([0.0])
        print(f"[DATA]  SNR over symbols: mean={snr_data_per_symbol.mean():.2f} dB | "
              f"median={np.median(snr_data_per_symbol):.2f} dB | min={snr_data_per_symbol.min():.2f} dB")
        plot_snr_over_time(snr_data_per_symbol, title="Data SNR over OFDM symbols")
    if plot and plot_opt['data_constellation'] and ground_truth:
        fig.tight_layout()
        plt.show()
    if plot and plot_opt.get('evm_vs_sub_carr', False) and ground_truth:
        plt.figure(); plt.plot(DATA_BINS, 20*np.log10(np.maximum(evm_avg,1e-12))); plt.title("EVM over subcarriers"); plt.show()

    # === 聚合 LLR，进入 LDPC 解码 ===
    all_llrs = np.concatenate(llr_blocks) if llr_blocks else np.array([], dtype=np.float32)
    src_idx = np.concatenate(src_idx_blocks) if src_idx_blocks else np.array([], dtype=np.int32)
    sub_carr_freq = np.concatenate(sub_carr_freq_blocks) if sub_carr_freq_blocks else np.array([], dtype=np.float32)

    c = ldpc.code(standard=LDPC_STANDARD, rate=LDPC_RATE, z=LDPC_Z, ptype=LDPC_PTYPE)
    K = c.K; Ncw = c.N
    total_blocks_avail = all_llrs.size // Ncw
    if total_blocks_avail == 0:
        print("No data LLRs collected; aborting.")
        return
    llrs_mat = all_llrs[:total_blocks_avail * Ncw].reshape(total_blocks_avail, Ncw)

    # --- 工具：批量解码若干块，返回 (信息比特列表, preBER, postBER) ---
    def decode_llrs_with_logging(
            c, llrs_used, *, K, Ncw,
            ground_truth: bool,
            gt_cw: np.ndarray | None = None,  # (nblocks, Ncw) 的真值码字；仅在 ground_truth=True 时可传
            print_limit: int = 10,
            batch: int = 512,
            microbatch: int | None = 256,
            dgl_device: str = "cuda",
            dgl_llr_clip: float = 20.0,
            dgl_max_iter: int = 200,
            dgl_verbose: bool = False,
            dgl_log_every: int = 1,
            dgl_check_every: int = 1,
    ):
        """
        批量解码 + 打印日志（ground_truth=True/False都支持）。
        返回:
          decoded_info: List[np.ndarray]  每块的信息比特（长度 K）
          stats: dict   汇总统计信息（均值/中位数等）
        """
        # ---- DGL/解码器参数（若 ldpc 实现不支持这些属性，设置也不会出错） ----
        try:
            c.dgl_device = dgl_device
            c.dgl_llr_clip = float(dgl_llr_clip)
            c.dgl_max_iter = int(dgl_max_iter)
            c.dgl_verbose = bool(dgl_verbose)
            c.dgl_log_every = int(dgl_log_every)
            c.dgl_check_every = int(dgl_check_every)
            if microbatch is not None:
                c.dgl_microbatch = int(microbatch)
        except Exception:
            pass

        # ---- 预处理 llrs_used 的形状 ----
        if isinstance(llrs_used, list):
            nblocks = len(llrs_used)
        else:
            nblocks = int(llrs_used.shape[0])

        decoded_info = []
        it_list = []
        pre_list = []
        post_list = []
        flip_rate_list = []
        llr_gain_list = []

        # ---- 主循环：分批解码 ----
        for s in range(0, nblocks, batch):
            e = min(s + batch, nblocks)

            # 组装 (b, Ncw)
            if isinstance(llrs_used, list):
                ch_batch = np.stack([llrs_used[i].copy() for i in range(s, e)], axis=0)
            else:
                ch_batch = llrs_used[s:e].copy()

            # 预判（硬判），作为 preBER / flip_rate 的“前状态”
            hard_pre = (ch_batch < 0).astype(np.uint8)  # (b, Ncw)

            # ground_truth=True 可计算 preBER
            if ground_truth and (gt_cw is not None):
                pre_ber = np.mean(hard_pre != gt_cw[s:e], axis=1)  # (b,)
            else:
                pre_ber = np.full((e - s,), np.nan, dtype=float)

            # 调用解码器（GPU/CPU）
            appB, itB = c.decode(ch_batch, dectype='sumprod2_dgl')  # appB: (b, Ncw)
            xhatB = (appB < 0).astype(np.uint8)  # (b, Ncw)

            # 统计：postBER 或替代指标
            if ground_truth and (gt_cw is not None):
                post_ber = np.mean(xhatB != gt_cw[s:e], axis=1)  # (b,)
                flip_rate = np.mean(hard_pre != xhatB, axis=1)  # “被纠正”的比例（可做参考）
                # LLR 置信度提升
                in_med = np.median(np.abs(ch_batch), axis=1)
                out_med = np.median(np.abs(appB), axis=1)
                llr_gain = np.where(in_med > 0, out_med / in_med, 1.0)
            else:
                post_ber = np.full((e - s,), np.nan, dtype=float)
                flip_rate = np.mean(hard_pre != xhatB, axis=1)  # 无GT时可报告 flip_rate
                in_med = np.median(np.abs(ch_batch), axis=1)
                out_med = np.median(np.abs(appB), axis=1)
                llr_gain = np.where(in_med > 0, out_med / in_med, 1.0)

            # 打印：按你的要求，“在 ground_truth=False 的时候进行打印”
            for j, i in enumerate(range(s, e)):
                # 迭代轮数：可能返回标量或每块不同；统一转 int
                it_show = int(itB if np.isscalar(itB) else itB[j])
                if (not ground_truth) and (i < print_limit):
                    print(f"[BLK {i:04d}] it={it_show:3d} flip_rate={flip_rate[j]:.4f} llr_gain={llr_gain[j]:.2f}")
                elif ground_truth and (i < print_limit):
                    # 有GT时也可打印（你也可以关掉）
                    print(f"[BLK {i:04d}] it={it_show:3d} preBER={pre_ber[j]:.4f} postBER={post_ber[j]:.4f} "
                          f"flip_rate={flip_rate[j]:.4f} llr_gain={llr_gain[j]:.2f}")

            # 收集
            for j in range(e - s):
                decoded_info.append(xhatB[j, :K])
            it_list.extend([int(itB if np.isscalar(itB) else itB[j]) for j in range(e - s)])
            pre_list.extend(pre_ber.tolist())
            post_list.extend(post_ber.tolist())
            flip_rate_list.extend(flip_rate.tolist())
            llr_gain_list.extend(llr_gain.tolist())

        if plot and plot_opt['BER_show'] and ground_truth:
            plot_pre_post_ber(pre_list, post_list)

        stats = {
            "it_mean": float(np.nanmean(it_list)),
            "it_median": float(np.nanmedian(it_list)),
            "preBER_mean": float(np.nanmean(pre_list)),
            "postBER_mean": float(np.nanmean(post_list)),
            "flip_rate_mean": float(np.mean(flip_rate_list)),
            "llr_gain_median": float(np.median(llr_gain_list)),
            "nblocks": nblocks,
        }
        return decoded_info, stats

    # === 第一步：先解出覆盖 64bit 头的最小块数 ===
    header_blocks = min(total_blocks_avail, math.ceil(HEADER_BITS / K))
    gt_cw = maybe_gt_cw(emit_bits_ldpc, 0, header_blocks, Ncw, ground_truth)
    decoded_info_h, stats_h = decode_llrs_with_logging(
        c, llrs_mat[:header_blocks],
        K=K, Ncw=Ncw,
        ground_truth=ground_truth,
        gt_cw=gt_cw,
    )
    print("[Head Stats]:")
    for key, value in stats_h.items():
        print(f"  - {key}: {value}")
    info_bits_concat = np.concatenate(decoded_info_h).astype(np.uint8)

    # 头部位于解码后的“信息比特”中，需要先解扰再解析
    info_bits_descr = scrambler_random(info_bits_concat, seed=SCRAMBLE_SEED)
    if info_bits_descr.size < HEADER_BITS:
        print(f"[ERR] 解出的信息比特不足 {HEADER_BITS}，got={info_bits_descr.size}")
        return

    header_bits = info_bits_descr[:HEADER_BITS]
    payload_len = u64_from_bits_msb(header_bits)
    print(f"[RX] header payload_len(bits)={payload_len}")

    # === 第二步：按头指出的长度，解出刚好需要的块数 ===
    total_info_needed = HEADER_BITS + payload_len
    need_blocks = min(total_blocks_avail, math.ceil(total_info_needed / K))

    decoded_info = decoded_info_h
    gt_cw2 = maybe_gt_cw(emit_bits_ldpc, header_blocks, need_blocks, Ncw, ground_truth)
    if need_blocks > header_blocks:
        decoded_info_2, stats_2 = decode_llrs_with_logging(
            c, llrs_mat[header_blocks:need_blocks],
            K=K, Ncw=Ncw,
            ground_truth=ground_truth,
            gt_cw=gt_cw2,
        )
        decoded_info.extend(decoded_info_2)
        print("[Data Stats]:")
        for key, value in stats_2.items():
            print(f"  - {key}: {value}")

    # 汇总所需的信息比特并解扰
    info_bits_all = np.concatenate(decoded_info)[:need_blocks * K].astype(np.uint8)
    info_bits_all = info_bits_all[:total_info_needed]  # 精确到头+payload
    info_bits_all_descr = scrambler_random(info_bits_all, seed=SCRAMBLE_SEED)

    # 取出 payload
    payload_bits = info_bits_all_descr[HEADER_BITS:HEADER_BITS + payload_len]
    print(f"[RX] got payload_bits={payload_bits.size}")

    # （可选）仅测试模式下做 BER 对比
    if ground_truth:
        try:
            gt_payload = get_bits_from_file(TXT_INPUT_PATH)  # 或 TIFF_INPUT_PATH
            if gt_payload.size >= payload_bits.size:
                print(f"[TX] got original file_bits={gt_payload.size}")
                gt_payload = gt_payload[:payload_bits.size]
                ber_payload = 1 - np.mean(payload_bits == gt_payload)
                print(f"[Payload BER] {ber_payload * 100:.4f}% (对齐文件真实长度)")
        except Exception:
            pass

    # 保存输出（按需要改名）
    out_bytes = np.packbits(payload_bits)
    with open(os.path.join(output_dir, "payload.bin"), 'wb') as f:
        f.write(out_bytes.tobytes())
    print(f"[RX] payload saved: {os.path.join(output_dir,'payload.bin')}  ({out_bytes.size} bytes)")

if __name__ == "__main__":
    assert os.path.exists(project_dir), "specify your proj dir"
    for dir_name in [output_dir, record_dir, data_dir]:
        os.makedirs(dir_name, exist_ok=True)
    plot_opt ={
        'correlation':                          True,
        'impulse_response':                     True,
        'raw_pilot_constellation':              True,
        'corrected_pilot_constellation':        True,
        'data_constellation':                   True,
        'unwrap':                               True,
        'received_signal':                      True,
        'evm_vs_sub_carr':                      True,
        'BER_show':                             True,
        'snr_time_pilot':                       True,
        'snr_time_data':                        True,
        'snr_over_sc':                          True,
    }
    analysis_txt(plot=True , plot_opt=plot_opt ,ground_truth=False ,tx_bits_path=TXT_INPUT_PATH)
