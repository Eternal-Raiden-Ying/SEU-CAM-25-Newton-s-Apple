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
TIFF_INPUT_PATH = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\data\fig1.tiff"

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
    x_auto, auto_unwrapped_phase, meta = phase_unwrap_auto(data=phase_shift, penal_bound=2e-4*(symbol_len*(-2*np.pi)/N)*gap)
    slope, intercept = fitting_line(x=x_auto, y=auto_unwrapped_phase, filter=True, residual_th=1.5)

    # plot_unwrap_phase_fitting_zoom(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N,window=100)
    # plot_unwrap_phase_fitting(n,phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N)

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

def maybe_gt_cw(emit_bits_ldpc, start_block: int, end_block: int, Ncw: int, ground_truth: bool):
    """
    返回 [start_block, end_block) 的真值码字矩阵 (nblocks, Ncw)。
    - ground_truth=False 或 emit_bits_ldpc is None → 返回 None
    - 安全：nblocks 以切片后的“实际可用比特数”计算，避免超大 pad
    """
    if (not ground_truth) or (emit_bits_ldpc is None):
        return None

    start_block = int(start_block)
    end_block   = int(end_block)
    Ncw         = int(Ncw)
    if end_block <= start_block:
        return np.zeros((0, Ncw), dtype=np.uint8)

    # 块号 → 比特号切片
    start_bit = start_block * Ncw
    end_bit   = end_block   * Ncw
    stream    = np.asarray(emit_bits_ldpc, dtype=np.uint8).ravel()

    # 实际可用比特（切片会自动裁边，不会越界）
    stream_slice = stream[start_bit:end_bit]

    # 以“实际可用比特数”为准计算 nblocks，且不超过请求的块数
    nblocks_req  = end_block - start_block
    nblocks_have = int(stream_slice.size // Ncw)           # 只用完整码字
    nblocks      = int(min(nblocks_req, nblocks_have))

    if nblocks <= 0:
        return np.zeros((0, Ncw), dtype=np.uint8)

    # 只取恰好 nblocks*Ncw 个比特，确保 reshape 对齐
    stream_slice = stream_slice[: nblocks * Ncw]
    return build_gt_codewords(stream_slice, nblocks, Ncw)

def build_gt_codewords(emit_bits_ldpc: np.ndarray, nblocks: int, Ncw: int) -> np.ndarray:
    """
    把编码后的比特流切成 (nblocks, Ncw) 的“真值码字矩阵”。
    若最后一个码字不满，则右侧 0 填充到最近的码字边界。
    重要：这里假定传入的是已按比特维切好的片段。
    """
    nblocks = int(nblocks)
    Ncw     = int(Ncw)
    if nblocks <= 0:
        return np.zeros((0, Ncw), dtype=np.uint8)

    stream = np.asarray(emit_bits_ldpc, dtype=np.uint8).ravel()
    total  = nblocks * Ncw

    if stream.size < total:
        # 仅补到“目标总长”，最多一个码字长度的差距；若大于一个码字，说明上游切片有误
        pad_len = total - stream.size
        if pad_len > Ncw:
            # 保护：将 pad 限制在单码字范围，并打印告警，避免异常超大内存
            print(f"[WARN] build_gt_codewords: pad_len({pad_len}) > Ncw({Ncw}). "
                  f"Clamping to at most one codeword. Check maybe_gt_cw slicing.")
            pad_len = ((stream.size + Ncw - 1) // Ncw) * Ncw - stream.size
            pad_len = max(pad_len, 0)

        if pad_len > 0:
            stream = np.pad(stream, (0, pad_len), constant_values=0)

    elif stream.size > total:
        # 多出的比特（理论上不会发生），裁掉即可
        stream = stream[:total]

    return stream.reshape(nblocks, Ncw)

def analysis_txt(plot=False, plot_opt=None, ground_truth=True, tx_bits_path=TXT_INPUT_PATH):
    """
    重排为 5 步（Step0~Step4；Step5 为可选图表已内嵌在各步中）：
      Step0 前处理/信道估计与分段
      Step1 解头：最少 LLR → 解 64-bit 头
      Step2 截断：依据 payload_len 计算 need_blocks / 需要的 data OFDM 数，裁剪数据范围
      Step3 主循环：在裁剪范围内继续均衡→PLL→(统计用 pre-MMSE)→MMSE→LLR，直至凑齐 need_blocks
      Step4 解码/解扰：一次性 LDPC 解码 → 截断 → 解扰 → 输出 payload
    ground_truth=False 时，不访问任何 emit_* / 源文件。
    """
    # -------------------------------
    # === Step0: 前处理 / 起始导频 ===
    # -------------------------------
    rx = np.load(
        fr"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\record\LDPC\received_txt_xjy_trival_1.npy")
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

    # 提取前 NUM_INIT_PILOTS 个导频符号
    rx_pilot = rx[ofdm_start: ofdm_start + num_symbols * (N + cp_len)]
    effective_symbols_num = num_symbols

    # 频域信道估计（起始连续导频）
    symbols = get_symbols(record=np.array(rx_pilot), cp_len=cp_len, N=N)
    H_fs = []
    for i in range(effective_symbols_num):
        H_f = evaluate_H_f(known_symbols=symbols[i, :], pilot_signals=pilot[i])
        H_fs.append(H_f)
    H_fs = H_fs[0:effective_symbols_num]

    # 估计采样漂移/常相位步进
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

    # 用起始导频外推得到 origin_H_f（频响平均并补偿漂移）
    origin_H_f = 0
    for index, H_f in enumerate(H_fs):
        origin_H_f += correct_H_f(H_f, delta, N, -index, symbol_len, fixed_phase_shift_factor=-fixed_phase_shift_factor)
    origin_H_f /= len(H_fs)

    if plot and plot_opt['impulse_response']:
        plot_impulse_response(np.fft.ifft(origin_H_f), fs)

    # Pilot SNR 粗评估（保留原逻辑）
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

    # 提取数据区（导频之后）
    print("---------------------------------------------------------------")
    print("Start to extract data part")
    delay = (np.floor(delta * effective_symbols_num * symbol_len).astype(np.int32)) + 1
    if plot and plot_opt['received_signal']:
        plot_received_signal(rx, ofdm_start, num_symbols, N, cp_len, delay)

    # ========== 提取所有后续 OFDM ==========
    rx_data = rx[ofdm_start + num_symbols * (N + cp_len):]
    symbols_all = get_symbols(record=rx_data, N=N, cp_len=cp_len)

    # ========== 基础索引 ==========
    M = symbols_all.shape[0]
    pilot_pos = np.arange(COMB_ITERATION, M, COMB_ITERATION + 1, dtype=int)
    data_pos = np.setdiff1d(np.arange(M), pilot_pos)

    # 统一容器（后续 Step1/Step3 共用）
    llr_blocks = []
    src_idx_blocks = []
    sub_carr_freq_blocks = []
    evm_accum = np.zeros(DATA_BINS.size, dtype=float)
    count_accum = np.zeros(DATA_BINS.size, dtype=int)
    snr_data_per_symbol = []
    snr_sc_sum = np.zeros(DATA_BINS.size)
    snr_sc_cnt = 0

    # 初始化 CPE PLL
    cpe_pll = DD_CPE_PLL(alpha=0.15, snr_th_db=6.0)

    # ======== 绘图缓存：保存 “PLL后、MMSE前” 的星座，键为全局 OFDM 索引 ========
    plot_cache = {}

    # ========== LDPC 参数与头部需求 ==========
    c = ldpc.code(standard=LDPC_STANDARD, rate=LDPC_RATE, z=LDPC_Z, ptype=LDPC_PTYPE)
    K = c.K
    Ncw = c.N

    per_ofdm_llrs = DATA_BINS.size * 2
    header_blocks = int(math.ceil(HEADER_BITS / K))
    need_llrs_head = header_blocks * Ncw
    # —— Step1 用到的 data 符号索引（全局 OFDM 索引）
    head_syms_used = []

    # ========== Step1：最小化 LLR → 先解 64-bit 头 ==========
    # 仅用“起始连续导频”的最后一块作为段起点（不预估整帧 comb-pilot）
    def _process_symbol_head(i: int):
        """仅用于 Step1：最小化开销拿到首个 LDPC block 的 LLR。
        - 只用起始连续导频的末符号 H_fs[-1] + (delta, fixed_phase_shift_factor) 外推
        - 不做 PLL，不做 MMSE 收缩，不作图
        """
        # 从“连续导频末”为起点外推（段起点索引视作 -1）
        dt = i - (-1)
        H_used = correct_H_f(
            H_fs[-1], delta, N, dt, symbol_len,
            fixed_phase_shift_factor=fixed_phase_shift_factor
        )

        # 均衡（直接得到 data 子载波的星座点）
        const = get_constellation(
            symbols=symbols_all[i, :], H_f=H_used,
            approximation=non_approximate, symbol_len=N
        )
        const = np.asarray(const).ravel()[DATA_START:-DATA_TAIL]

        # 直接以星座统计的 MAD 估计生成 LLR（QPSK）
        llr_sym, st = qpsk_llrs_from_constellation(
            const, llr_clip=20.0, k_sigma_clip=4.0, belief_scale=0.5, use_mad=True
        )

        # 轻量级稳健缩放（避免极端噪声时 LLR 过激）
        evm2 = 0.5 * (st['sigma_r'] ** 2 + st['sigma_i'] ** 2)
        scale = np.clip(1.0 / (1.0 + 6.0 * evm2), 0.6, 1.2)

        # 记录（注意 per-OFDM 输出是 2*|DATA_BINS| 个 LLR）
        llr_blocks.append(llr_sym * scale)
        src_idx_blocks.append(np.full(llr_sym.size, i, dtype=np.int32))
        sub_carr_freq_blocks.append(np.repeat(np.linspace(0, fs, N)[DATA_BINS], 2))
        # ... 省略：生成 llr_sym 并 append 到 llr_blocks 等
        head_syms_used.append(i)  # 记录该 data 符号已用于 Step1（避免 Step3 重复处理）
        return llr_sym.size

    # 逐个 data OFDM 产出 LLR，直到湊够“首个 LDPC block”所需的 Ncw 个 LLR 即停
    produced_llrs = 0
    for i in data_pos.tolist():
        produced_llrs += _process_symbol_head(i)
        if produced_llrs >= need_llrs_head:  # need_llrs_head = header_blocks * Ncw，通常=1个block
            break

    # 聚合并解头（解码函数保持与原版一致）
    all_llrs_so_far = np.concatenate(llr_blocks) if llr_blocks else np.array([], dtype=np.float32)
    total_blocks_avail_so_far = all_llrs_so_far.size // Ncw
    if total_blocks_avail_so_far == 0:
        print("[ERR] Not enough LLRs to decode header.")
        return
    llrs_mat_head = all_llrs_so_far[:header_blocks * Ncw].reshape(header_blocks, Ncw)

    def decode_llrs_with_logging(
            c, llrs_used, *, K, Ncw, ground_truth: bool,
            gt_cw: np.ndarray | None = None, print_limit: int = 200,
            batch: int = 512, microbatch: int | None = 256,
            dgl_device: str = "cuda", dgl_llr_clip: float = 20.0,
            dgl_max_iter: int = 200, dgl_verbose: bool = False,
            dgl_log_every: int = 1, dgl_check_every: int = 1,
    ):
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

        if isinstance(llrs_used, list):
            nblocks = len(llrs_used)
        else:
            nblocks = int(llrs_used.shape[0])

        decoded_info, it_list, pre_list, post_list = [], [], [], []
        flip_rate_list, llr_gain_list = [], []
        for s in range(0, nblocks, batch):
            e = min(s + batch, nblocks)
            ch_batch = (np.stack([llrs_used[i].copy() for i in range(s, e)], axis=0)
                        if isinstance(llrs_used, list) else llrs_used[s:e].copy())
            hard_pre = (ch_batch < 0).astype(np.uint8)

            if ground_truth and (gt_cw is not None):
                pre_ber = np.mean(hard_pre != gt_cw[s:e], axis=1)
            else:
                pre_ber = np.full((e - s,), np.nan, dtype=float)

            appB, itB = c.decode(ch_batch, dectype='sumprod2_dgl')
            xhatB = (appB < 0).astype(np.uint8)

            if ground_truth and (gt_cw is not None):
                post_ber = np.mean(xhatB != gt_cw[s:e], axis=1)
                flip_rate = np.mean(hard_pre != xhatB, axis=1)
                in_med = np.median(np.abs(ch_batch), axis=1)
                out_med = np.median(np.abs(appB), axis=1)
                llr_gain = np.where(in_med > 0, out_med / in_med, 1.0)
            else:
                post_ber = np.full((e - s,), np.nan, dtype=float)
                flip_rate = np.mean(hard_pre != xhatB, axis=1)
                in_med = np.median(np.abs(ch_batch), axis=1)
                out_med = np.median(np.abs(appB), axis=1)
                llr_gain = np.where(in_med > 0, out_med / in_med, 1.0)

            for j, i_b in enumerate(range(s, e)):
                it_show = int(itB if np.isscalar(itB) else itB[j])
                if (not ground_truth) and (i_b < print_limit):
                    print(f"[BLK {i_b:04d}] it={it_show:3d} flip_rate={flip_rate[j]:.4f} llr_gain={llr_gain[j]:.2f}")
                elif ground_truth and (i_b < print_limit):
                    print(f"[BLK {i_b:04d}] it={it_show:3d} preBER={pre_ber[j]:.4f} postBER={post_ber[j]:.4f} "
                          f"flip_rate={flip_rate[j]:.4f} llr_gain={llr_gain[j]:.2f}")

            decoded_info.extend([xhat[:K] for xhat in xhatB])
            it_list.extend([int(itB if np.isscalar(itB) else itB[j]) for j in range(e - s)])
            pre_list.extend(pre_ber.tolist());
            post_list.extend(post_ber.tolist())
            flip_rate_list.extend(flip_rate.tolist());
            llr_gain_list.extend(llr_gain.tolist())

        if plot and plot_opt.get('BER_show', False) and ground_truth:
            plot_pre_post_ber(np.array(pre_list), np.array(post_list))

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

    gt_cw_head = maybe_gt_cw(None, 0, 0, Ncw, False)  # Step1 不用 GT
    decoded_info_h, stats_h = decode_llrs_with_logging(
        c, llrs_mat_head, K=K, Ncw=Ncw, ground_truth=False, gt_cw=gt_cw_head
    )
    print("[Head Stats]:")
    for k, v in stats_h.items():
        print(f"  - {k}: {v}")

    info_bits_concat = np.concatenate(decoded_info_h).astype(np.uint8)
    info_bits_descr = scrambler_random(info_bits_concat, seed=SCRAMBLE_SEED)
    if info_bits_descr.size < HEADER_BITS:
        print(f"[ERR] 解出的信息比特不足 {HEADER_BITS}，got={info_bits_descr.size}")
        return
    header_bits = info_bits_descr[:HEADER_BITS]
    payload_len = u64_from_bits_msb(header_bits)
    # payload_len = 641968
    print(f"[RX] header payload_len(bits)={payload_len}")


    # ---- Header 合理性校验 + 一次重试 + 最终夹紧 ----
    # 理论容量上限（基于当前录音中可见的 data OFDM 总数）
    max_blocks_possible_all = int((len(data_pos) * per_ofdm_llrs) // Ncw)
    max_payload_bits_upper  = max(0, max_blocks_possible_all * K - HEADER_BITS)

    def _header_invalid(pl):
        return (pl < 0) or (pl > max_payload_bits_upper)

    if _header_invalid(payload_len):
        print(f"[WARN] Implausible header payload_len={payload_len} "
              f"(capacity={max_payload_bits_upper} bits). Retrying header decode with +2 OFDM...")

        # —— 追加 2 个 data OFDM 的 LLR 再解一次头
        head_ofdm_used = int(math.ceil(produced_llrs / per_ofdm_llrs))
        extra = min(2, max(0, len(data_pos) - head_ofdm_used))
        for _ in range(extra):
            nxt = int(data_pos[head_ofdm_used])
            produced_llrs += _process_symbol_head(nxt)
            head_ofdm_used += 1

        all_llrs_so_far = np.concatenate(llr_blocks)
        llrs_mat_head = all_llrs_so_far[:header_blocks * Ncw].reshape(header_blocks, Ncw)
        decoded_info_h, stats_h = decode_llrs_with_logging(
            c, llrs_mat_head, K=K, Ncw=Ncw, ground_truth=False, gt_cw=None
        )
        info_bits_concat = np.concatenate(decoded_info_h).astype(np.uint8)
        info_bits_descr  = scrambler_random(info_bits_concat, seed=SCRAMBLE_SEED)
        header_bits      = info_bits_descr[:HEADER_BITS]
        payload_len_retry = u64_from_bits_msb(header_bits)
        print(f"[RX] header payload_len_retry(bits)={payload_len_retry}")

        if _header_invalid(payload_len_retry):
            print(f"[WARN] Header still implausible after retry. "
                  f"Clamping payload_len to capacity={max_payload_bits_upper}.")
            payload_len = max_payload_bits_upper
        else:
            payload_len = payload_len_retry


    # ========== Step2：依据头 → 计算 need_blocks/OFDM 并截断 ==========
    total_info_needed = HEADER_BITS + payload_len
    need_blocks = int(math.ceil(total_info_needed / K))
    need_llrs_total = need_blocks * Ncw
    need_data_ofdm = int(math.ceil(need_llrs_total / per_ofdm_llrs))
    need_data_ofdm = min(need_data_ofdm, len(data_pos))
    if need_data_ofdm <= 0:
        print("[ERR] No data OFDM available after header.")
        return
    last_symbol_idx = int(data_pos[need_data_ofdm - 1])

    # —— 截断到真实帧末尾
    M_trunc = last_symbol_idx + 1
    symbols_all = symbols_all[:M_trunc]
    pilot_pos = pilot_pos[pilot_pos <= last_symbol_idx]
    data_pos = data_pos[data_pos <= last_symbol_idx]
    symbols_comb_pilot = symbols_all[pilot_pos]

    # —— 截断后再估计 comb-pilot 与分段参数（与你原逻辑一致）
    Hf_comb = []
    for j, comb_p in enumerate(symbols_comb_pilot):
        pilot_ref = generate_pilot_symbol(N, seed=COMB_PILOT_SEED_BASE + j)
        Hf_p = evaluate_H_f(known_symbols=comb_p, pilot_signals=pilot_ref)
        Hf_comb.append(Hf_p)
    Hf_comb = np.array(Hf_comb, dtype=complex) if len(Hf_comb) > 0 else np.zeros((0, N), dtype=complex)

    pilot_qualities = []
    seg_params = []

    H_ref = H_fs[-1]  # 段起点参考（初始为连续导频外推回的起点）
    prev_ref_idx = -1  # 段起点全局OFDM索引（连续导频末视为 -1）
    last_delta = float(delta)  # 记录最近一次段内拟合的漂移斜率（供尾段复用）
    last_phi = float(fixed_phase_shift_factor)

    if len(Hf_comb) > 0:
        for j, pidx in enumerate(pilot_pos):
            # --- (1) 用上一参考 + 全局漂移外推到当前 comb 位置：H_pred（用于质量评估）
            dt_pred = int(pidx - prev_ref_idx)
            H_pred = correct_H_f(
                H_ref, delta, N, dt_pred, symbol_len,
                fixed_phase_shift_factor=fixed_phase_shift_factor
            )

            # 质量评估：用“非同源”的 H_pred 去均衡当前 comb，再与已知导频比较
            comb_p = symbols_comb_pilot[j]
            const_p = get_constellation(
                symbols=comb_p, H_f=H_pred,
                approximation=non_approximate, symbol_len=N
            )
            const_p = np.asarray(const_p).ravel()[DATA_START:-DATA_TAIL]
            ref_p = generate_pilot_symbol(N, seed=COMB_PILOT_SEED_BASE + j)[DATA_BINS]
            q, _ = pilot_quality_from_constellation(const_p, ref_p)
            pilot_qualities.append(q)

            # --- (2) 段内漂移拟合：上一参考 H_ref -> 当前 comb 的单符号估计 Hf_comb[j]
            gap = int(pidx - prev_ref_idx)
            delta_j, phi_step_j = fit_drift_between(
                H_ref, Hf_comb[j], gap, N=N, symbol_len=symbol_len, return_phi=True
            )
            seg_params.append((prev_ref_idx, int(pidx), H_ref, delta_j, phi_step_j))
            last_delta, last_phi = float(delta_j), float(phi_step_j)

            # --- (3) 推进参考到“当前 comb”，供下一段使用
            H_ref = Hf_comb[j].copy()
            prev_ref_idx = int(pidx)

        # 尾段：从最后一个 comb 到帧末（M_trunc），沿用最近一段的漂移参数
        if prev_ref_idx < M_trunc - 1:
            seg_params.append((prev_ref_idx, int(M_trunc), H_ref, last_delta, last_phi))

        pilot_qualities = np.asarray(pilot_qualities, dtype=float)

    else:
        # 没有 comb-pilot：整段用连续导频的外推参数
        pilot_qualities = np.asarray([], dtype=float)
        seg_params = [(-1, int(M_trunc), H_fs[-1], float(delta), float(fixed_phase_shift_factor))]

    # ========== （可选）GT 参考，仅用于统计/作图，不参与长度决策 ==========
    emit_bits = emit_bits_ldpc = emit_constellations = None
    if ground_truth:
        emit_bits, emit_bits_ldpc, emit_constellations, _ = _build_gt(
            tx_bits_path, data_bins_size=DATA_BINS.size, scramble_seed=SCRAMBLE_SEED
        )

    # ========== 绘图准备（基于截断后的 data_pos） ==========
    if plot and plot_opt.get('data_constellation', False) and ground_truth:
        n_rows, n_cols = 4, 4
        pic_num = n_rows * n_cols
        pic_idx = np.linspace(start=0,
                              stop=0 + data_pos.size // (pic_num - 1) * (pic_num - 1),
                              num=pic_num).astype(np.int32)
        pic_idx = data_pos[pic_idx]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))

    # === （仅 GT 用于作图/统计）索引映射初始化 ===
    gt_enabled = bool(ground_truth and (emit_constellations is not None))
    data_index_map = {}  # { 全局OFDM索引 i -> emit_constellations 的行号 }

    # —— 关键：GT 行映射从 “Step1 已用的 data 符号数量” 起步，避免参考错位
    gt_next_row = len(head_syms_used)

    # —— 关键：Step3 跳过 Step1 已处理过的 data 符号，避免 LLR 重复追加
    processed_set = set(head_syms_used)

    # ========== Step3：主循环（在截断范围内，补齐到 need_blocks） ==========
    def _process_one_symbol(i: int):
        """完整处理（含 comb 融合），用于 Step3。返回该符号产生的 LLR 数。"""
        # —— 声明会在本函数内修改的外层变量（自增/赋值）
        nonlocal gt_next_row, snr_sc_cnt

        # 找到所属分段
        s_idx2, e_idx2, H_start, delta_j, phi_step_j = None, None, None, None, None
        for (s, e, Hs, dj, phj) in seg_params:
            if (s + 1) <= i <= (e - 1):
                s_idx2, e_idx2, H_start, delta_j, phi_step_j = s, e, Hs, dj, phj
                break
        if s_idx2 is None:
            s_idx2, e_idx2, H_start, delta_j, phi_step_j = seg_params[0]

        # 1) 起点外推
        dt = i - s_idx2
        H_from_start = correct_H_f(H_start, delta_j, N, dt, symbol_len, fixed_phase_shift_factor=phi_step_j)

        # 2) 最近 comb 外推 + 融合
        if len(pilot_pos) > 0:
            j_near = int(np.argmin(np.abs(pilot_pos - i)))
            idx_near = pilot_pos[j_near]
            H_near = Hf_comb[j_near]
            dt2 = i - idx_near
            H_from_near = correct_H_f(H_near, delta_j, N, dt2, symbol_len, fixed_phase_shift_factor=phi_step_j)
            if (pilot_qualities is not None) and (len(pilot_qualities) > 0):
                q_start = pilot_qualities[max(j_near - 1, 0)]
                q_near = pilot_qualities[j_near]
                w1, w2 = blend_weights_quality(i, s_idx2, idx_near, q_start=q_start, q_near=q_near)
            else:
                w1, w2 = _blend_weights(i, s_idx2, idx_near, mode=CHAN_BLEND_MODE, weights=CHAN_BLEND_WEIGHTS)
        else:
            H_from_near = H_from_start
            w1, w2 = _blend_weights(i, s_idx2, s_idx2, mode=CHAN_BLEND_MODE, weights=CHAN_BLEND_WEIGHTS)

        H_used = (w1 * H_from_start + w2 * H_from_near) / (w1 + w2 + 1e-12)

        # 3) 均衡
        const = get_constellation(symbols=symbols_all[i, :], H_f=H_used,
                                  approximation=non_approximate, symbol_len=N)
        const = np.asarray(const).ravel()[DATA_START:-DATA_TAIL]

        # 4) 统计参考（pre-MMSE）→ PLL
        if gt_enabled:
            # 首次触达该 data 符号 i 时，按顺序映射到 emit_constellations 的下一行
            if (i not in data_index_map) and (gt_next_row < emit_constellations.shape[0]):
                data_index_map[i] = gt_next_row
                gt_next_row += 1

            if i in data_index_map:
                ref_sc = emit_constellations[data_index_map[i]].ravel()
            else:
                # 安全兜底：行数不够或未映射，退化为硬判参考
                ref_sc = (np.sign(const.real) + 1j * np.sign(const.imag)) / np.sqrt(2)
        else:
            ref_sc = (np.sign(const.real) + 1j * np.sign(const.imag)) / np.sqrt(2)

        snr_med_for_pll = float(np.median(evm_and_snr(const, ref_sc)[2]))
        const = cpe_pll.step(const, snr_med_db=snr_med_for_pll)

        # —— 绘图缓存：PLL后、MMSE前
        if plot and plot_opt.get('data_constellation', False) and gt_enabled:
            plot_cache[i] = const.copy()

        # 5) MMSE 收缩（LLR仍基于收缩后的星座）
        _, st_tmp = qpsk_llrs_from_constellation(const, use_mad=True)
        H_used_bins = H_used[DATA_BINS]
        const = apply_mmse_shrinkage(const, H_used_bins, st_tmp['sigma_r'], st_tmp['sigma_i'])

        # 6) 统计（仅 GT 有参考时）
        if gt_enabled and (i in data_index_map):
            ref_sc = emit_constellations[data_index_map[i]].ravel()
            evm_rms_sym, _, snr_db_sym = evm_and_snr(const, ref_sc)
            snr_data_per_symbol.append(float(np.median(snr_db_sym)))
            snr_sc_sum[:] += snr_db_sym
            snr_sc_cnt += 1
            w_sc = llr_scale_from_snr(snr_db_sym, lo=2.0, hi=10.0, min_scale=0.4, max_scale=1.0)
        else:
            w_sc = np.ones(DATA_BINS.size, dtype=float)

        # 7) LLR
        llr_sym, st = qpsk_llrs_from_constellation(
            const, llr_clip=20.0, k_sigma_clip=4.0, belief_scale=0.5, use_mad=True
        )
        evm2 = 0.5 * (st['sigma_r'] ** 2 + st['sigma_i'] ** 2)
        scale = np.clip(1.0 / (1.0 + 6.0 * evm2), 0.6, 1.2)
        llr_sym_2col = llr_sym.reshape(-1, 2)
        llr_sym = (llr_sym_2col * w_sc[:, None]).ravel()

        llr_blocks.append(llr_sym * scale)
        src_idx_blocks.append(np.full(llr_sym.size, i, dtype=np.int32))
        sub_carr_freq_blocks.append(np.repeat(np.linspace(0, fs, N)[DATA_BINS], 2))

        return llr_sym.size

    # —— 继续遍历截断范围内的 data 符号，直到块数达到 need_blocks
    def _blocks_collected():
        return (np.sum([b.size for b in llr_blocks]) // Ncw) if llr_blocks else 0

    for (s_idx2, e_idx2, _, _, _) in seg_params:
        for i in range(s_idx2 + 1, min(e_idx2 - 1, M_trunc - 1) + 1):
            if i not in data_pos:
                continue
            if i in processed_set:  # ★ 避免重复处理 Step1 用过的符号
                continue

            _ = _process_one_symbol(i)

            # 绘图（使用缓存的 pre-MMSE 星座）
            if plot and plot_opt.get('data_constellation', False) and ground_truth and ('pic_idx' in locals()) \
                    and (i in plot_cache) and (i in pic_idx.tolist()) and (emit_constellations is not None) \
                    and (i in data_index_map):
                seq = int(np.where(pic_idx == i)[0][0])
                ax = axes[seq // n_cols, seq % n_cols]
                draw_constellation_map(received=plot_cache[i],
                                       emit_pilot=emit_constellations[data_index_map[i]],
                                       title=f'constellation{i}', ax=ax, limit_border=2)

            if _blocks_collected() >= need_blocks:
                break
        if _blocks_collected() >= need_blocks:
            break

    if plot and plot_opt.get('data_constellation', False) and ground_truth and 'fig' in locals():
        fig.tight_layout();
        plt.show()
    if plot and plot_opt.get('snr_time_data', False) and ground_truth and len(snr_data_per_symbol) > 0:
        snr_data_per_symbol = np.array(snr_data_per_symbol, dtype=float)
        print(f"[DATA] SNR over symbols: mean={snr_data_per_symbol.mean():.2f} dB | "
              f"median={np.median(snr_data_per_symbol):.2f} dB | min={snr_data_per_symbol.min():.2f} dB")
        plot_snr_over_time(snr_data_per_symbol, title="Data SNR over OFDM symbols")

    # ========== Step4：统一解码 / 解扰 / 保存 ==========
    all_llrs = np.concatenate(llr_blocks) if llr_blocks else np.array([], dtype=np.float32)
    total_blocks_avail = all_llrs.size // Ncw
    if total_blocks_avail == 0:
        print("No data LLRs collected; aborting.");
        return

    llrs_mat = all_llrs[:min(need_blocks, total_blocks_avail) * Ncw].reshape(
        min(need_blocks, total_blocks_avail), Ncw
    )

    # 头部已解（decoded_info_h），继续解剩余
    decoded_info = decoded_info_h
    if need_blocks > header_blocks:
        gt_cw_2 = maybe_gt_cw(emit_bits_ldpc, header_blocks, need_blocks, Ncw, ground_truth)
        decoded_info_2, stats_2 = decode_llrs_with_logging(
            c, llrs_mat[header_blocks:need_blocks], K=K, Ncw=Ncw,
            ground_truth=ground_truth, gt_cw=gt_cw_2
        )
        decoded_info.extend(decoded_info_2)
        print("[Data Stats]:")
        for k, v in stats_2.items():
            print(f"  - {k}: {v}")

    info_bits_all = np.concatenate(decoded_info)[:need_blocks * K].astype(np.uint8)
    info_bits_all = info_bits_all[:total_info_needed]
    info_bits_all_descr = scrambler_random(info_bits_all, seed=SCRAMBLE_SEED)

    payload_bits = info_bits_all_descr[HEADER_BITS:HEADER_BITS + payload_len]
    print(f"[RX] got payload_bits={payload_bits.size}")

    if ground_truth:
        try:
            gt_payload = get_bits_from_file(TXT_INPUT_PATH)
            if gt_payload.size >= payload_bits.size:
                print(f"[TX] got original file_bits={gt_payload.size}")
                gt_payload = gt_payload[:payload_bits.size]
                ber_payload = 1 - np.mean(payload_bits == gt_payload)
                print(f"[Payload BER] {ber_payload * 100:.4f}% (对齐文件真实长度)")
        except Exception:
            pass

    out_bytes = np.packbits(payload_bits)
    with open(os.path.join(output_dir, "payload.tiff"), 'wb') as f:
        f.write(out_bytes.tobytes())
    print(f"[RX] payload saved: {os.path.join(output_dir, 'payload.tiff')}  ({out_bytes.size} bytes)")

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
    analysis_txt(plot=False , plot_opt=plot_opt ,ground_truth=True ,tx_bits_path=TIFF_INPUT_PATH)
