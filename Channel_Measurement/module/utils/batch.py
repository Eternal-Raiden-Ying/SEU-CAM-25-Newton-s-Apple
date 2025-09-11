# -*- coding: utf-8 -*-
"""
utils/batch.py
- _qpsk_hard: 把星座就近硬判到标准 QPSK 点
- QPSK_reflection: 把星座直接转换为 bit（支持 clockwise，默认 False；仅关键字）
- analyze_pilots: 前导/comb 导频质量分析（向量化），使用 _qpsk_hard + QPSK_reflection
- evaluate_H_f / correct_H_f / correct_H_f_batch / estimate_drift_and_origin
- build_segments_from_pilots: 利用前导+comb 构建每个数据符号的参考与权重
- get_constellation / dd_cpe_pll_apply / robust_sigma / mmse_shrinkage / llr_from_constellation
- llr_scale_by_snr / pack_llr_blocks / pll_snr_median / generate_comb_pilot_symbol
- synchronize: 基于 scipy.signal.correlate 的时域同步
- ldpc_* / bits_from_file / scramble_bits: 解码与扰码 I/O（保持接口）
"""

from __future__ import annotations

import warnings

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.signal import correlate as sp_correlate
from .math_process import phase_unwrap_auto, fitting_line, segment_means_on
from .ldpc_jossy import code
from .demodulate import _qpsk_hard, QPSK_reflection, get_constellation



# ========= EVM / SNR =========
def evm_from_constellation(const: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    逐子载波 EVM^2（功率归一），对齐最后一维；返回同形状。
    """
    const = np.asarray(const)
    ref = np.asarray(ref)
    error = const - ref
    return (np.abs(error) ** 2) / (np.abs(ref) ** 2 + 1e-12)

def snr_from_constellation(const: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    逐子载波 SNR（线性），对齐最后一维；返回同形状。
    """
    evm2 = evm_from_constellation(const, ref)
    return 1.0 / np.maximum(evm2, 1e-12)

def _mad_sigma(x: np.ndarray, axis=-1):
    """MAD 估计标准差：sigma ≈ 1.4826 * median(|x - median(x)|)."""
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=False)
    return 1.4826 * mad

def _esno_from_sigmas(sig_r: float, sig_i: float):
    """
    利用 qpsk_llrs_from_constellation 的 robust sigma 估计 Es/N0。
    对 QPSK（单位能量 Es=1）有：Es/N0 = 1 / (2 * σ^2),
    其中 σ^2 取实部虚部的平均方差。
    """
    sigma2 = 0.5 * (sig_r**2 + sig_i**2)
    esno_lin = 1.0 / (2.0 * sigma2 + 1e-12)
    esno_db  = 10.0 * np.log10(esno_lin)
    return esno_lin, esno_db

# ========= Comb 导频生成 =========
def _generate_comb_pilot_symbol(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = N // 2
    real_parts = rng.choice([-1, 1], size=half - 1)
    imag_parts = rng.choice([-1, 1], size=half - 1)
    X_half = (real_parts + 1j * imag_parts) / np.sqrt(2)
    X = np.zeros(N, dtype=complex)
    X[0] = 1
    X[1:half] = X_half
    X[half] = 1
    X[half + 1:] = np.conj(X_half[::-1])
    return X

def generate_comb_pilot_symbol(N: int, seed: int) -> np.ndarray:
    return _generate_comb_pilot_symbol(N, seed)

# ========= H(f) 估计 / 外推 =========
def evaluate_H_f(symbols_td: np.ndarray,
                 pilots_fd: np.ndarray | None,
                 DATA_BINS: np.ndarray | None = None) -> np.ndarray:
    """
    统一版 H(f) 估计：支持 1D 或 2D。
      - symbols_td: [N] 或 [ns, N]（时域，已去 CP）
      - pilots_fd : 同维度；为 None 时需提供 seeds（deprecated）
    返回与 symbols_td 对齐。
    """
    X = np.asarray(symbols_td)
    assert pilots_fd is not None
    eps = 1e-6
    pilots_fd = np.where(pilots_fd==0.0, eps, pilots_fd)
    if X.ndim == 1:
        Yf = np.fft.fft(X)
        if DATA_BINS is None:
            DATA_BINS = np.arange(X.size)
        if pilots_fd is None:
            raise RuntimeError("Deprecated function invoked, pilots_fd must be explicitly given")
        if pilots_fd.size > DATA_BINS.size:
            pilots_fd = pilots_fd[DATA_BINS]
        Y_f_used = Yf[DATA_BINS]
        H_f = np.full(X.shape, np.nan, dtype=np.complex128)
        H_f[DATA_BINS] = Y_f_used / np.asarray(pilots_fd)
        return H_f
    elif X.ndim == 2:
        ns, N = X.shape
        Yf = np.fft.fft(X, axis=1)
        if DATA_BINS is None:
            DATA_BINS = np.arange(N)
        if pilots_fd is None:
            raise RuntimeError("Deprecated function invoked, pilots_fd must be explicitly given")
        if pilots_fd.shape[1] > DATA_BINS.size:
            pilots_fd = pilots_fd[:,DATA_BINS]
        Y_f_used = Yf[:,DATA_BINS]
        H_fs = np.full(X.shape, np.nan, dtype=np.complex128)
        H_fs[:,DATA_BINS] = Y_f_used / np.asarray(pilots_fd)
        return H_fs
    else:
        raise ValueError("symbols_td 维度必须为 1 或 2")

def correct_H_f(origin_H_f: np.ndarray,
                N: int,
                index: Optional[int, np.ndarray],
                symbol_len: int,
                delta: Optional[float, np.ndarray],
                fixed_phase_shift_factor: Optional[float, np.ndarray] = 0.0) -> np.ndarray:
    k = np.linspace(-N//2, N//2, N, endpoint=False, dtype=np.int32)
    k = np.concatenate([k[N//2:], k[:N//2]])
    if isinstance(index, np.ndarray) and index.size > 1:
        if origin_H_f.ndim > 1:
            k, _ = np.broadcast_arrays(k, origin_H_f)
            index, _ = np.broadcast_arrays(index.reshape(-1, 1), origin_H_f)
        else:
            index, origin_H_f = np.broadcast_arrays(index.reshape(-1,1), origin_H_f.reshape(1,-1))
            k, _ = np.broadcast_arrays(k, origin_H_f)
        if isinstance(delta, np.ndarray):
            delta = delta.reshape(-1,1)
        if isinstance(fixed_phase_shift_factor, np.ndarray):
            fixed_phase_shift_factor = fixed_phase_shift_factor.reshape(-1,1)
    linear_phase = np.exp(-1j * 2*np.pi/N * (delta * index * symbol_len) * k)
    cpe = np.exp(1j * fixed_phase_shift_factor * index)
    return origin_H_f * linear_phase * cpe


def estimate_drift_and_origin(Hf_seq: np.ndarray, *, N: int, symbol_len: int, return_plot_args: bool=False, mode='each'):
    """
    输入 [ns, N] 的 H(f) 序列（前导 pilot），拟合 (delta, phi_step)，并把所有 H 对齐求均值得到 origin_Hf。
    """
    H = np.asarray(Hf_seq)
    assert H.ndim == 2
    ratios = H[1::1]/H[:-1:1]
    xs, phases, slopes, intercepts, deltas, phis = [], [], [], [], [0], [0]
    for ratio in ratios:
        x_auto, auto_unwrapped_phase, _ = phase_unwrap_auto(data=ratio)
        slope, intercept = fitting_line(x=x_auto, y=auto_unwrapped_phase, filter=True, residual_th=1.2)
        xs.append(x_auto.copy())
        phases.append(auto_unwrapped_phase.copy())
        slopes.append(slope)
        intercepts.append(intercept)
        deltas.append(slope / (symbol_len * (-2 * np.pi) / N))
        phis.append(intercept)
    origin = list()
    num_pilot = H.shape[0]
    for idx in range(num_pilot):
        origin.append(correct_H_f(
            origin_H_f=H[idx],
            delta=np.sum(deltas[:idx+1]),
            N=N,
            index=-1,
            symbol_len=symbol_len,
            fixed_phase_shift_factor=np.sum(phis[:idx+1])
        ))
    origin = np.array(origin)

    h_abs = np.abs(origin)
    h_mean = np.mean(h_abs, axis=0)
    h_std = np.std(h_abs, axis=0)
    mask = np.where(h_abs < h_mean[:None] + h_std[:None], 1, 0)
    w = np.where(mask, mask.shape[0]/np.sum(mask, axis=0), 0)
    # origin = np.average(origin, axis=0,weights=w)
    origin = np.average(origin, axis=0)

    if mode == 'total':
        ratio = np.mean(H[1:]/H[:-1], axis=0)
        x_auto, auto_unwrapped_phase, _ = phase_unwrap_auto(data=ratio)
        slope, intercept = fitting_line(x=x_auto, y=auto_unwrapped_phase, filter=True, residual_th=1.2)
        plot_args = {
            'ratio': ratio,
            'slope': np.array(slope),
            'intercept': np.array(intercept),
            'x_auto': np.array(x_auto),
            'auto_unwrapped_phase': np.array(auto_unwrapped_phase),
            'N': N
        }
        if not return_plot_args:
            return np.sum(deltas).astype(float)/(num_pilot-1), np.sum(phis).astype(float)/(num_pilot-1), origin
        else:
            return np.sum(deltas).astype(float)/(num_pilot-1), np.sum(phis).astype(float)/(num_pilot-1), origin, plot_args
    elif mode == 'each':
        plot_args = {
            'ratio': ratios,
            'slope': np.array(slopes),
            'intercept': np.array(intercepts),
            'x_auto': np.array(xs),
            'auto_unwrapped_phase': np.array(phases),
            'N': N
        }
        if not return_plot_args:
            return np.array(deltas).astype(float), np.array(phis).astype(float), origin
        else:
            return np.array(deltas).astype(float), np.array(phis).astype(float), origin, plot_args
    else:
        raise ValueError(f"unknown mode {mode}")


# ========= 段构建 =========

def _fit_drift_between(H_start, H_end, gap, N, * ,
                       symbol_len=None, return_phi=False,
                       plot: bool | int = False, DATA_BINS: np.ndarray | None = None):
    """
    用两个时间点（相隔 gap 个 OFDM）的信道估计做比值，拟合得到“每 OFDM”的
    频偏斜率 delta 以及常相位步进 phi。
    """
    """
    由相隔 gap 个 OFDM 的两次信道估计，拟合得到：
      - delta：每符号的线性相位斜率（对应 SFO/CFO 残差）
      - phi_step：每符号公共相位步进（CPE）
    """
    if symbol_len is None:
        raise ValueError("symbol_len must be provided")
    if DATA_BINS is None:
        DATA_BINS = np.arange(N)
    if H_start.ndim == 1:
        phase_shift = H_end[DATA_BINS] * np.conj(H_start[DATA_BINS])
    else:
        raise ValueError(f"only support dimension <= 1, received {H_start.ndim}")
    x_auto, auto_unwrapped_phase, meta = phase_unwrap_auto(data=phase_shift.flatten(), DATA_BINS=DATA_BINS, N=N)
    slope, intercept = fitting_line(x=x_auto, y=auto_unwrapped_phase, filter=True, residual_th=1.5)
    if type(plot) is bool and plot:
        from .plot import plot_unwrap_phase_fitting
        plot_unwrap_phase_fitting(phase_shift, slope, intercept,x_auto, auto_unwrapped_phase, N)
    elif type(plot) is int:
        from .plot import plot_unwrap_phase_fitting
        plot_unwrap_phase_fitting(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N, title=f'pilot {plot}')

    delta = slope / (gap * symbol_len * (-2 * np.pi) / N)
    phi_step = intercept / gap  # 每“一个”符号的常相位步进

    if return_phi:
        return float(delta), float(phi_step)
    return float(delta)


def _pilot_quality(symbol_td: np.ndarray,
                   H_ref_start: np.ndarray,
                   *,
                   delta: float,
                   phi_step: float,
                   gap: int,
                   pilot_ref_fd: np.ndarray,
                   DATA_BINS: np.ndarray,
                   N: int,
                   symbol_len: int) -> tuple[float, float]:
    """
    用“上一段参考 H_ref_start”外推到本 pilot 处的 H_pred，再据此评估 pilot 质量。
    返回：(q, snr_med_db)

    q 的定义沿用之前：对 median SNR(dB) 做一个 S 形压缩，范围 (0,1)。
    """
    # 1) 预测本 pilot 位置的 H(f)
    H_pred = correct_H_f(
        origin_H_f=H_ref_start, N=N, index=gap, symbol_len=symbol_len,
        delta=delta, fixed_phase_shift_factor=phi_step
    )

    # 2) 用 H_pred 等化该 pilot 的星座
    Xd = get_constellation(symbols_td=symbol_td, H_used=H_pred, DATA_BINS=DATA_BINS)
    Rd = pilot_ref_fd[DATA_BINS] if pilot_ref_fd.size == N else pilot_ref_fd

    # 3) SNR / 质量
    snr_sc = snr_from_constellation(Xd, Rd)                         # per-SC 线性 SNR
    snr_db_med = float(10.0 * np.log10(np.median(np.clip(snr_sc, 1e-12, None))))
    q = 1.0 / (1.0 + np.exp(-(snr_db_med - 5.0) / 2.0))             # 压缩到 (0,1)

    return q, snr_db_med

def build_segments_from_pilots(H_start: np.ndarray,
                               Hf_comb: np.ndarray,
                               pilot_pos: np.ndarray,
                               M: int,
                               DATA_BINS: np.ndarray,
                               q_comb: Optional[np.ndarray] = None,
                               *,
                               mode: str = "quality_distance",
                               data_pos: np.ndarray | None = None,
                               start_idx: int | None = None,
                               symbol_len: int,
                               N: int,
                               fs: float = 48000,
                               delta_guard_th: float = 1e-4,
                               # 全局漂移（无 comb 或尾段回退时使用）
                               delta_global: float = 0.0,
                               phi_global: float = 0.0,
                               # 若未提供 q_comb，则用这两项内部计算 q
                               symbols_comb_td: Optional[np.ndarray] = None,
                               pilot_ref_comb_fd: Optional[np.ndarray] = None,
                               # 权重控制
                               distance_power: float = 1.0,
                               eps: float = 1.0) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[np.ndarray, float]]]:
    """
        构建“段（segment）级”的信道参考与漂移参数，并为每个数据 OFDM 符号给出两路参考的
        外推间隔（dt）与融合权重（w1, w2）。本函数还可在未显式提供 comb 质量 q 时，内部评估
        comb 的 pilot 质量，并将其纳入权重计算。(此时需要传入symbols_comb_td以及pilot_ref_comb_fd)

        v2: 输出便于计算comb导频的预测漂移信道的参数，包括 H_start, delta, phi, gap

        段的定义与拟合：
            - 以“前导导频序列”的最后一块信道估计 H_start 为初始参考，索引视作 -1。
            - 给定 comb 导频在 0..M-1 区间内的 OFDM 索引 `pilot_pos`，将区间划分为：
                (-1, p1], (p1, p2], ..., (plast, M]
            - 每个分段 (start_idx, end_idx]，用上一参考 H_ref 与右端 comb 的 H 做比值，调用
                `fit_drift_between(H_ref, H_comb, gap=end_idx-start_idx, ...)` 拟合得到
                每符号线性相位斜率 delta 与每符号 CPE 步进 phi_step（两者均为该段常数）。
            - 段内每个数据 OFDM i（i ∈ (start_idx, end_idx]）：
                dt_from_start = i - start_idx
                dt_from_near  = i - end_idx           # 注意可能为负数；权重时取绝对值
                “起点参考”使用 (H_start, delta, phi_step, dt_from_start) 外推；
                “近端参考”使用 (H_near,  delta, phi_step, dt_from_near ) 外推，其中 H_near
                为右端 comb 的 H，若无 comb 则与 H_start 相同。

        质量与权重：
            - 质量 q 的来源优先级：
                (1) 若传入 q_comb（与 comb 一一对应）→ 直接使用；
                (2) 否则在 symbols_comb_td + pilot_ref_comb_fd 可用时，内部调用
                    `_pilot_quality(...)`：用上一段参考 H_ref 及拟合的 (delta, phi, gap)
                    外推得到 H_pred，对该 comb pilot 进行“非同源”等化评估，得到 q；
                (3) 否则 q = 1（仅按距离）。
            - 对段内每个数据符号 i，计算距离 d1 = dt_from_start，d2 = |dt_from_near|，
                以“距离×质量”的规则得到两路权重并归一化：
                    w1 ∝ 1 / (d1 + eps)^distance_power
                    w2 ∝ q / (d2 + eps)^distance_power
                返回的 w1_per_sym, w2_per_sym 满足逐样本相加为 1。
            - 若无 comb（Hf_comb 为空）或尾段回退，使用全局漂移 (delta_global, phi_global)。

        兜底处理：
            - 正常情况下，所有数据符号都会被某一段覆盖；如出现边界/索引异常导致仍有剩余
                （即 ptr < n_data），则用“最后一段”的参数为剩余样本填充，并设置 w1=1, w2=0，
                只使用起点参考，避免在异常情况下的权重混合引入不稳定。

    参数：
        H_start           : np.ndarray [N]                    前导最后一块的 H(f)（频域）
        Hf_comb           : np.ndarray [n_comb, N]            comb 导频处的 H(f)
        pilot_pos         : np.ndarray [n_comb]               comb 导频在 0..M-1 区间内的 OFDM 索引
        M                 : int                               本批次（数据+comb）总符号数，定义区间右端
        DATA_BINS         : np.ndarray [Nd]                   有效子载波索引（内部只在质量评估时使用）
        q_comb            : Optional[np.ndarray [n_comb]]     已有的 comb 质量（0..1），可选
        mode              : str                               预留，目前使用 "quality_distance"
        symbol_len        : int                               OFDM 符号长度（含 CP）
        N                 : int                               FFT 点数
        delta_global      : float                             无 comb/尾段回退时的全局每符号相位斜率
        phi_global        : float                             无 comb/尾段回退时的全局每符号 CPE 步进
        symbols_comb_td   : Optional[np.ndarray [n_comb, N]]  comb 的时域符号（去 CP）
        pilot_ref_comb_fd : Optional[np.ndarray [n_comb, N]]  comb 的频域参考
        distance_power    : float                             距离权重幂次，>1 更强调近端
        eps               : float                             距离平滑正则，防止分母为 0

    返回：
        dict，键及形状：
        "H_start"                : [n_data, N]   每个comb导频符号对应段的起点参考 H
        "delta"                  : [n_data]      相位斜率 delta
        "phi"                    : [n_data]      符号 CPE 步进 phi
        "gap"                    : [n_data]      到起点的符号间隔 gap

        dict，键及形状：
        "H_start_per_seg"        : [n_data, N]   每个数据符号对应段的起点参考 H
        "delta_per_seg"          : [n_data]      段内每符号相位斜率 delta
        "phi_per_seg"            : [n_data]      段内每符号 CPE 步进 phi_step
        "dt_from_start_per_sym"  : [n_data]      到段起点的符号间隔 dt1
        "H_near_per_sym"         : [n_data, N]   段右端 comb 的参考 H（无 comb 时与起点相同）
        "delta_per_seg_per_sym"  : [n_data]      与上面相同（为兼容原有接口保留）
        "phi_per_seg_per_sym"    : [n_data]      同上
        "dt_from_near_per_sym"   : [n_data]      到右端 comb 的符号间隔 dt2（可能为负；权重取绝对值）
        "w1_per_sym"             : [n_data]      起点参考权重
        "w2_per_sym"             : [n_data]      近端参考权重
        之后可据此构造两路外推并做加权：
        H_used = w1 * correct_H_f_batch(H_start_per_seg, delta, phi, dt1, ...) \
               + w2 * correct_H_f_batch(H_near_per_sym,  delta, phi, dt2, ...)
    """

    all_idx  = np.arange(M, dtype=int)
    pilot_pos = np.asarray(pilot_pos, dtype=int)
    data_pos = np.setdiff1d(all_idx, pilot_pos) if data_pos is None else data_pos
    n_data   = data_pos.size

    # ------- 1) 组段（与之前一致） -------
    segs = []
    prev_idx = -1 if start_idx is None else start_idx
    H_ref = H_start.copy()

    pb_idx = [10,43,109]
    valid_pilot_list = [-1]
    H_s_list, d_list, p_list, g_list = [], [], [], []
    if Hf_comb.size > 0 and pilot_pos.size > 0:
        for j, pidx in enumerate(pilot_pos):
            gap = int(pidx - prev_idx)
            d_j, p_j = _fit_drift_between(H_ref, Hf_comb[j], gap, N=N, symbol_len=symbol_len,
                                          return_phi=True, DATA_BINS=DATA_BINS, plot=False)
            # if np.abs(d_j) > delta_guard_th:
            #     continue
            valid_pilot_list.append(pidx)
            H_s_list.append(H_ref.copy())
            d_list.append(d_j)
            p_list.append(p_j)
            g_list.append(gap)
            segs.append((prev_idx, int(pidx), H_ref.copy(), Hf_comb[j].copy(), float(d_j), float(p_j), j, gap))
            prev_idx = int(pidx)
            H_ref = Hf_comb[j].copy()
        if prev_idx < M - 1:
            if segs:
                last_d, last_p = segs[-1][4], segs[-1][5]
            else:
                last_d, last_p = float(delta_global), float(phi_global)
            H_end = correct_H_f(origin_H_f=H_ref,N=N,
                                index=int(M - prev_idx),symbol_len=symbol_len,
                                delta=last_d,fixed_phase_shift_factor=last_p)
            segs.append((prev_idx, M, H_ref.copy(), H_end.copy(), float(last_d), float(last_p), len(pilot_pos)-1, int(M - prev_idx)))
    else:
        H_end = correct_H_f(origin_H_f=H_ref, N=N,
                            index=int(M - prev_idx), symbol_len=symbol_len,
                            delta=delta_global, fixed_phase_shift_factor=phi_global)
        segs.append((prev_idx, M, H_start.copy(), H_end.copy(), float(delta_global), float(phi_global), -1, int(M - prev_idx)))

    # freq_offset
    deltas = np.array(d_list)
    freq_offsets = fs/ (deltas+1) - fs
    ofdm_idx = np.array(valid_pilot_list)
    # here new_deltas[0] correspond to Hf[0]/Hf[-1]
    if ofdm_idx.size > 1:
        interp_freq_offset = segment_means_on(freq_offsets=freq_offsets.copy(), ofdm_idx=ofdm_idx,
                                              eval_idx=np.concatenate([np.array([-1]),all_idx]),
                                              smoothing=0.0, extrap='hold')
        new_deltas = 1 / (interp_freq_offset / fs + 1) - 1

        from matplotlib import pyplot as plt
        plt.plot(all_idx, interp_freq_offset)
        plt.scatter(all_idx - 0.5, interp_freq_offset, marker='*', color='red', label='interpolate')
        plt.scatter(np.arange(pilot_pos[-1] + 1) - 0.5, np.repeat(freq_offsets, np.diff(ofdm_idx)), marker='o', s=4,
                    color='black', label='comb pilot')
        for i in ofdm_idx:
            plt.axvline(i, linestyle='dotted', color='black')
        plt.axvline(M - 1, linestyle='dotted', color='black')
        plt.legend()
        plt.show()

    else:
        new_deltas = None

    # ------- 2) 准备每段的 q（若未传 q_comb 则内部计算） -------
    q_list = None
    if q_comb is not None:
        # 明确给了 q_comb：按老逻辑使用（长度应与 comb 数一致）
        q_list = np.asarray(q_comb).astype(float)
    elif (symbols_comb_td is not None) and (pilot_ref_comb_fd is not None) and (len(segs) > 0) and (pilot_pos.size > 0):
        # 内部评估 q：一段一个 q（用该段右端的 comb）
        q_vals = []
        for (start_idx, end_idx, Hs, Hn, d_j, p_j, j_idx, gap) in segs:
            if j_idx < 0 or j_idx >= pilot_pos.size:
                # 尾段沿用上一段的 q；若无上一段则 q=1
                if len(q_vals) > 0:
                    q_vals.append(q_vals[-1])
                else:
                    q_vals.append(1.0)
                continue
            q_j, _ = _pilot_quality(
                symbol_td=symbols_comb_td[j_idx],
                H_ref_start=Hs,
                delta=d_j, phi_step=p_j, gap=gap,
                pilot_ref_fd=pilot_ref_comb_fd[j_idx],
                DATA_BINS=DATA_BINS, N=N, symbol_len=symbol_len
            )
            q_vals.append(float(q_j))
        q_list = np.asarray(q_vals, dtype=float)
    # else: q_list 维持 None -> 仅按距离权重

    # ------- 3) 为每个数据符号映射段参数并计算权重 -------
    H_start_per  = np.empty((n_data, N), dtype=H_start.dtype)
    H_near_per   = np.empty((n_data, N), dtype=H_start.dtype)
    delta_from_s = np.empty(n_data, dtype=float)
    delta_from_e = np.empty(n_data, dtype=float)
    phi_per      = np.empty(n_data, dtype=float)
    dt1          = np.empty(n_data, dtype=float)
    dt2          = np.empty(n_data, dtype=float)
    w1           = np.empty(n_data, dtype=float)
    w2           = np.empty(n_data, dtype=float)

    ptr = 0
    for (start_idx, end_idx, Hs, Hn, d_j, p_j, j_idx, gap) in segs:
        mask = (data_pos > start_idx) & (data_pos <= end_idx)
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        i_vals = data_pos[idxs].astype(float)

        dt_from_start = i_vals - float(start_idx)
        near_idx = float(end_idx)
        dt_from_near  = i_vals - near_idx
        d1 = np.maximum(dt_from_start, 0.0)
        d2 = np.abs(dt_from_near)

        # 计算comb到data的平均delta, new_deltas中的delta是相邻symbol的
        if new_deltas is not None:
            if pilot_pos[-1] < M-1 and end_idx == M:
                delta_used_mask = np.concatenate([i_vals.astype(int)])
                delta_used = new_deltas[delta_used_mask]
                delta_from_start = np.array([np.sum(delta_used[:i]) / i for i in range(1, delta_used.size+1)])
                delta_from_end = np.repeat([d_j], i_vals.size)
            else:
                delta_used_mask = np.concatenate([i_vals.astype(int), np.array([end_idx])])
                delta_used = new_deltas[delta_used_mask]
                delta_from_start = np.array([np.sum(delta_used[:i])/i for i in range(1,delta_used.size)])
                delta_from_end = np.array([np.sum(delta_used[-i:])/i for i in range(1, delta_used.size)])[::-1]

        # 质量因子：本段对应的 comb 质量（若不可得则为 1）
        if q_list is not None and j_idx is not None and j_idx >= 0 and j_idx < q_list.size:
            qj = float(np.clip(q_list[j_idx], 1e-3, 1.0))
        else:
            qj = 1.0

        w1_seg = 1.0 / np.power(d1 + eps, distance_power)
        w2_seg = qj * (1.0 / np.power(d2 + eps, distance_power))
        s = (w1_seg + w2_seg)
        w1_seg = w1_seg / s
        w2_seg = w2_seg / s

        cnt = idxs.size
        H_start_per[ptr:ptr+cnt, :] = Hs[None, :]
        H_near_per[ptr:ptr+cnt,  :] = Hn[None, :]
        delta_from_s[ptr:ptr+cnt]   = delta_from_start if new_deltas is not None else d_j
        delta_from_e[ptr:ptr+cnt]   = delta_from_end if new_deltas is not None else d_j
        phi_per[ptr:ptr+cnt]        = p_j
        dt1[ptr:ptr+cnt]            = dt_from_start
        dt2[ptr:ptr+cnt]            = dt_from_near
        w1[ptr:ptr+cnt]             = w1_seg
        w2[ptr:ptr+cnt]             = w2_seg
        ptr += cnt

    if ptr < n_data:
        raise RuntimeError("Unexpected Error! check the logic in segments building")

    pilot_seg = {
        'H_start': np.array(H_s_list),
        'delta': np.array(d_list),
        'phi': np.array(p_list),
        'gap': np.array(g_list)
    }
    data_seg = {
        "H_start_per_seg":           H_start_per,
        "delta_per_seg":             delta_from_s,
        "phi_per_seg":               phi_per,
        "dt_from_start_per_sym":     dt1,

        "H_near_per_sym":            H_near_per,
        "delta_per_seg_per_sym":     delta_from_e,
        "phi_per_seg_per_sym":       phi_per,
        "dt_from_near_per_sym":      dt2,

        "w1_per_sym":                w1,
        "w2_per_sym":                w2,
    }

    return pilot_seg, data_seg



# ========= Pilot/Comb 统一分析（向量化）=========
def analyze_pilots(symbols_td: np.ndarray | None,
                   Hf: np.ndarray | None,
                   pilot_ref: np.ndarray,
                   DATA_BINS: np.ndarray,
                   *,
                   mode: str = "front",
                   clockwise: bool = False,
                   symbols_fd: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """
        分析前导/comb 导频在“非同源等化”条件下的质量与误码指标。
        典型用法：对每个导频符号，传入该导频的时域符号 `symbols_td`（或已准备好的频域符号 `symbols_fd`），
        以及用于等化的“非同源”信道估计/预测 `Hf`（来自上一段参考外推等），并以频域参考 `pilot_ref`
        作为理想星座，计算 SNR（中位数）、质量分数 q、MAD 估计的 σ_r/σ_i、以及 BER/SER。

        功能概述：
            1) 若提供 `symbols_fd`，直接在 DATA_BINS 上取子带作为 Xd；
             否则执行 Xf = FFT(symbols_td) / Hf，再在 DATA_BINS 上取子带得到 Xd。
            2) 用参考星座 Rd = pilot_ref[:, DATA_BINS] 计算逐子载波 SNR（基于 EVM），并取中位数(dB)。
            3) 将 SNR(dB) 通过 sigmoid 压缩为质量分数 q ∈ (0,1)。
            4) 对 Xd 做 QPSK 硬判（_qpsk_hard），以误差的 MAD 估计 σ_r/σ_i，并换算 Es/N0（_esno_from_sigmas）。
            5) 使用 QPSK_reflection 提取比特，与参考比特比较，得 BER 与 SER。

        参数：
        symbols_td : np.ndarray，形状 [ns, N]
                     时域导频 OFDM 符号（已去 CP）。当 `symbols_fd` 提供时可为占位，不参与计算。
        Hf         : np.ndarray，形状 [ns, N]
                     对应导频的“非同源”信道估计/预测（用于等化，避免同源高估质量）。
                     若传入与该导频同源的 Hf（由同一导频估计），则 q/SNR 可能被乐观估计。
        pilot_ref  : np.ndarray，形状 [ns, N]
                     频域参考导频（理想星座，逐符号一帧），用于计算 SNR/EVM 与 BER/SER。
        DATA_BINS  : np.ndarray，形状 [Nd]
                     有效子载波索引；结果在这些子载波上统计。
        mode       : str，默认 "front"
                     标记来源（如 "front"/"comb"），本函数内部不分支，仅作上层区分用途。
        clockwise  : bool，默认 False
                     QPSK 位映射方向，传给 QPSK_reflection，需与发端保持一致。
        symbols_fd : Optional[np.ndarray]，形状 [ns, N]
                     可选，若已在外部完成 FFT/等化，传入希望直接评估的频域符号序列；
                     函数将直接在 DATA_BINS 上切片：Xd = symbols_fd[:, DATA_BINS]，不再使用 `Hf` 参与等化。

        返回：
        dict[str, np.ndarray]，各字段均为长度 ns 的一维数组：
            - "snr_db_med" : 每符号的 SNR 中位数（dB），由 per-SC EVM 推得后取中位数。
            - "esno_db"    : 由误差 MAD 估计的 Es/N0（dB），基于 σ_r/σ_i。
            - "q"          : 质量分数 ∈ (0,1)，定义为 sigmoid((snr_db_med - 5) / 2)。
            - "sigma_r"    : 误差在 I 轴上的 MAD 标准差估计。
            - "sigma_i"    : 误差在 Q 轴上的 MAD 标准差估计。
            - "ber"        : 基于 QPSK_reflection 的比特误码率（每两个比特为一符号）。
            - "ser"        : 符号误码率（对每两个比特聚合后任一位错即计错）。

        注意：
            - 为得到有意义的 q/SNR，请确保传入的 Hf 为“非同源”预测（例如由上一参考外推到本导频处的 H_pred），
            而非用本导频自身估计出来的 Hf；同源等化会过度乐观。
            - 若你已外部完成 FFT/等化并通过 `symbols_fd` 传入，本函数不会再使用 `Hf` 做除法，请确保
            `symbols_fd` 的语义与 Rd 一致（即与 pilot_ref 同步对齐）。
    """
    pilot_ref = np.asarray(pilot_ref)
    if pilot_ref.ndim == 1:
        pilot_ref = pilot_ref[None,:]
    Rd = pilot_ref[:, DATA_BINS] if pilot_ref.shape[1]>DATA_BINS.size else pilot_ref
    if symbols_fd is None:
        assert symbols_td is not None and Hf is not None
        symbols_td = np.asarray(symbols_td)
        Hf = np.asarray(Hf)
        Xd = get_constellation(symbols_td=symbols_td, H_used=Hf, DATA_BINS=DATA_BINS)
    else:
        if symbols_fd.ndim == 1:
            symbols_fd = symbols_fd[None,:]
        Xd = symbols_fd[:,DATA_BINS] if symbols_fd.shape[1]>DATA_BINS.size else symbols_fd

    assert Xd.shape == Rd.shape
    snr_sc = snr_from_constellation(Xd, Rd)
    snr_db_med = 10.0 * np.log10(np.median(np.clip(snr_sc, 1e-12, None), axis=-1))
    quality = 1.0 / (1.0 + np.exp(-(snr_db_med - 5.0) / 2.0))  # 仅用于回退时

    hard = _qpsk_hard(Xd)
    err = Xd - hard
    sigma_r = _mad_sigma(err.real, axis=-1) + 1e-12
    sigma_i = _mad_sigma(err.imag, axis=-1) + 1e-12
    _, esno_db = _esno_from_sigmas(sigma_r, sigma_i)

    est_bits = QPSK_reflection(Xd, clockwise=clockwise)
    ref_bits = QPSK_reflection(Rd, clockwise=clockwise)
    diff = (est_bits != ref_bits).astype(np.uint8)
    ber = diff.mean(axis=-1)
    ser = diff.reshape(diff.shape[0], -1, 2).any(axis=-1).mean(axis=-1)

    return {
        "snr_db_med": snr_db_med.astype(float),
        "esno_db": esno_db.astype(float),
        "q": quality.astype(float),
        "sigma_r": sigma_r.astype(float),
        "sigma_i": sigma_i.astype(float),
        "ber": ber.astype(float),
        "ser": ser.astype(float)
    }


# ========= 等化 / PLL / 噪声 / 收缩 / LLR =========

def apply_cpe_pll_sequence(constellations: np.ndarray,
                           snr_med_db: np.ndarray,
                           *,
                           alpha: float = 0.15,
                           snr_th_db: float = 6.0,
                           alpha_min: float = 0.05,
                           alpha_max: float = 0.30,
                           snr_th_min_db: float = 3.0,
                           snr_th_max_db: float = 10.0,
                           beta: float = 0.9,
                           snr_mid_db: float = 6.0,
                           snr_scale: float = 4.0) -> np.ndarray:
    """
    逐符号应用 DD_CPE_PLL（单环），输入与输出形状均为 [n_sym, Nd]。
    """
    constellations = np.asarray(constellations)
    snr_med_db = np.asarray(snr_med_db).reshape(-1,)
    assert constellations.ndim == 2 and constellations.shape[0] == snr_med_db.size
    from .decoder_oop import PLLConfig, DD_CPE_PLL
    pll_cfg = PLLConfig(alpha=alpha, snr_th_db=snr_th_db,
                        alpha_min=alpha_min, alpha_max=alpha_max,
                        snr_th_min_db=snr_th_min_db, snr_th_max_db=snr_th_max_db,
                        beta=beta, snr_mid_db=snr_mid_db, snr_scale=snr_scale)
    pll = DD_CPE_PLL(pll_cfg)
    # pll = DD_CPE_PLL(alpha=alpha, snr_th_db=snr_th_db,
    #                  alpha_min=alpha_min, alpha_max=alpha_max,
    #                  snr_th_min_db=snr_th_min_db, snr_th_max_db=snr_th_max_db,
    #                  beta=beta, snr_mid_db=snr_mid_db, snr_scale=snr_scale)
    out = np.empty_like(constellations)
    for i in range(constellations.shape[0]):
        out[i] = pll.step(constellations[i], float(snr_med_db[i]))
    return out


def dd_cpe_pll_apply(constellations: np.ndarray,
                     snr_med_db: np.ndarray,
                     *, alpha: float = 0.15,
                     snr_th_db: float = 6.0) -> np.ndarray:
    """
    判决导向 CPE 一阶 PLL（向量化）：
      constellations: [n, Nd] or [Nd]
      snr_med_db:     [n]     or scalar
    """
    s = np.asarray(constellations)
    if s.ndim == 1:
        s2 = s[None, :]
        snr = np.asarray(snr_med_db).reshape(1,)
    else:
        s2 = s
        snr = np.asarray(snr_med_db).reshape(-1,)
    hard = _qpsk_hard(s2)
    num = np.sum(s2 * np.conj(hard), axis=1)          # [n]
    phi = -np.angle(num)                               # [n]
    mask = (snr >= snr_th_db).astype(float)[:, None]
    rot = np.exp(1j * (alpha * phi)[:, None] * mask)
    out = s2 * rot
    return out if s.ndim == 2 else out[0]

def robust_sigma(constellations: np.ndarray, *, per_sc: bool=False) -> Dict[str, np.ndarray]:
    s = np.asarray(constellations)
    if s.ndim == 1:
        s = s[None, :]
    hard = _qpsk_hard(s)
    err = s - hard
    if per_sc:
        sr = np.abs(err.real) * 1.2533
        si = np.abs(err.imag) * 1.2533
    else:
        sr = _mad_sigma(err.real, axis=-1) + 1e-12
        si = _mad_sigma(err.imag, axis=-1) + 1e-12
    return {"sigma_r": sr, "sigma_i": si}

def mmse_shrinkage(constellations: np.ndarray,
                   Habs2: np.ndarray,
                   sigmas: Dict[str, np.ndarray]) -> np.ndarray:
    """
    支持 const: [Nd] 或 [B, Nd]
         Habs2: 标量 / [Nd] / [B] / [B,Nd]
         sigma_r/sigma_i: 标量 / [Nd] / [B] / [B,Nd]
    """
    s = np.asarray(constellations)
    s2d = s.ndim == 2
    if not s2d:
        s = s[None, :]
    B, Nd = s.shape

    def to_2d(a):
        a = np.asarray(a)
        if a.ndim == 0:
            return np.full((B, Nd), float(a), dtype=float)
        if a.ndim == 1:
            if a.shape[0] == Nd:
                return np.broadcast_to(a[None, :], (B, Nd)).astype(float, copy=False)
            if a.shape[0] == B:
                return np.broadcast_to(a[:, None], (B, Nd)).astype(float, copy=False)
            raise ValueError(f"Shape {a.shape} incompatible with (B={B}, Nd={Nd})")
        if a.ndim == 2:
            if a.shape == (B, Nd):
                return a.astype(float, copy=False)
            if a.shape == (1, Nd):
                return np.broadcast_to(a, (B, Nd)).astype(float, copy=False)
            if a.shape == (B, 1):
                return np.broadcast_to(a, (B, Nd)).astype(float, copy=False)
            raise ValueError(f"Shape {a.shape} incompatible with (B,Nd)=({B},{Nd})")
        raise ValueError(f"Unsupported ndim={a.ndim}")

    H2   = to_2d(Habs2)
    sigR = to_2d(sigmas["sigma_r"])
    sigI = to_2d(sigmas["sigma_i"])
    N0   = 0.5 * (sigR**2 + sigI**2)

    H2   = np.clip(H2, 1e-12, None)
    N0   = np.clip(N0, 1e-12, None)
    shrink = H2 / (H2 + N0)

    out = s * shrink
    return out if s2d else out[0]


def llr_from_constellation(constellations: np.ndarray,
                           *, mod: str = "QPSK",
                           llr_clip: float = 20.0,
                           clockwise: bool = False) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    QPSK 近似 LLR（向量化，位序与 QPSK_reflection 保持一致）：
      - clockwise=False: LLR[bit0=bI] 用 Q 轴，LLR[bit1=bQ] 用 I 轴
      - clockwise=True : LLR[bit0=bI] 用 I 轴，LLR[bit1=bQ] 用 Q 轴
    返回 llr 形状 [n, Nd*2]（单符号时 [1, Nd*2]），以及每子载波的 SNR(dB) 估计字典。
    """
    s = np.asarray(constellations)
    squeeze_back = False
    if s.ndim == 1:
        s = s[None, :]
        squeeze_back = True

    # 估计每符号的轴向噪声方差（MAD -> σ，再平方）
    hard = _qpsk_hard(s)
    err = s - hard
    sig_r = _mad_sigma(err.real, axis=-1) + 1e-9
    sig_i = _mad_sigma(err.imag, axis=-1) + 1e-9
    var_r = (sig_r**2)[:, None]
    var_i = (sig_i**2)[:, None]

    # 轴向 LLR（BPSK 近似）：bit=0 ↔ 正半轴
    if clockwise:
        L0 = 2.0 * s.real / var_r  # bI ← I 轴
        L1 = 2.0 * s.imag / var_i  # bQ ← Q 轴
    else:
        L0 = 2.0 * s.imag / var_i  # bI ← Q 轴
        L1 = 2.0 * s.real / var_r  # bQ ← I 轴

    L0 = np.clip(L0, -llr_clip, llr_clip)
    L1 = np.clip(L1, -llr_clip, llr_clip)

    llr = np.stack([L0, L1], axis=-1).reshape(s.shape[0], -1).astype(np.float32)

    # 供外部缩放参考的 per-SC SNR(dB)
    ev = np.mean(np.abs(err)**2, axis=0, keepdims=True) + 1e-12   # [1, Nd]
    snr_db_per_sc = 10*np.log10(1.0/ev)
    return (llr if not squeeze_back else llr), {"snr_db_per_sc": np.repeat(snr_db_per_sc, s.shape[0], axis=0)}


def llr_scale_by_snr(snr_db_per_sc: np.ndarray,
                     *, lo: float = 2.0, hi: float = 10.0,
                     min_scale: float = 0.4, max_scale: float = 1.0) -> np.ndarray:
    x = np.clip((snr_db_per_sc - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    return min_scale + x * (max_scale - min_scale)

def pack_llr_blocks(ofdm_idx: np.ndarray,
                    sub_carr_freq: np.ndarray,
                    llr: np.ndarray,
                    Ncw: int):
    """
    输入:
        ofdm_idx        : [ns, Nd*2]  每个比特所属的 OFDM 符号序号
        sub_carr_freq   : [ns, Nd*2]  每个比特所属子载波频率(仅用于稳定排序/可视化)
        llr             : [ns, Nd*2]
    输出(dict):
        'llr'       : [num_blocks, code_N]  —— 原调用仅用它
        'ofdm_idx'  : [num_blocks, code_N]  —— 与 'llr' 一一对应
        'sc_freq'   : [num_blocks, code_N]  —— 与 'llr' 一一对应
    """
    n_codeword = llr.size // Ncw
    llr = llr.flatten()[:n_codeword*Ncw].reshape(-1, Ncw)
    nc, N = llr.shape
    assert ofdm_idx.size >= llr.size
    assert sub_carr_freq.size >= llr.size
    flat_ofdm = ofdm_idx.reshape(-1).astype(np.int32)[:nc*N]
    flat_freq = sub_carr_freq.reshape(-1).astype(np.float32)[:nc*N]

    return {
        'llr': llr,
        'ofdm_idx': flat_ofdm.reshape(nc, N),
        'sc_freq': flat_freq.reshape(nc, N)
    }


# ========= M 推断 & PLL SNR 中位 =========
def equalize_first_data_symbol(symbols_all_td: np.ndarray,
                               origin_Hf: np.ndarray, delta: float, phi_step: float,
                               *, N: int, DATA_BINS: np.ndarray, symbol_len: int = None) -> np.ndarray:
    if symbol_len is None:
        symbol_len = N
    H1 = correct_H_f(
        origin_H_f=origin_Hf,
        delta=delta,
        N=N,
        symbol_len=symbol_len,
        index=1,
        fixed_phase_shift_factor=phi_step
    )
    return get_constellation(symbols_td=symbols_all_td[0], H_used=H1, DATA_BINS=DATA_BINS)

def estimate_M_from_filesize(*, filesize_bytes: int, K: int, Ncw: int, Nd: int, modulation_bits: int, interval: int) -> int:
    info_bits = int(filesize_bytes) * 8
    n_codewords = (info_bits + K - 1) // K
    coded_bits = n_codewords * Ncw
    bits_per_ofdm = Nd * modulation_bits
    data_syms = (coded_bits + bits_per_ofdm - 1) // bits_per_ofdm
    comb_syms = np.ceil(data_syms/interval) - 1 if interval else 0
    return int(data_syms + comb_syms)

def pll_snr_median(const_zf: np.ndarray) -> np.ndarray:
    """
    由均衡后（未收缩）的星座计算每符号 SNR 中位数（dB），向量化：
      const_zf: [n, Nd] or [Nd]
      返回： [n] 或 标量
    """
    s = np.asarray(const_zf)
    if s.ndim == 1:
        ref = _qpsk_hard(s)
        snr_sc = snr_from_constellation(s, ref)                # [Nd]
        return 10.0 * np.log10(np.median(np.clip(snr_sc, 1e-12, None)))
    else:
        ref = _qpsk_hard(s)                                     # [n, Nd]
        snr_sc = snr_from_constellation(s, ref)                 # [n, Nd]
        return 10.0 * np.log10(np.median(np.clip(snr_sc, 1e-12, None), axis=-1))


def choose_next_pilots(
    data_pos: np.ndarray,
    available_pilots: np.ndarray,
    edge_expand_k: int = 1,
) -> np.ndarray:
    """
    从 available_pilots 中，按 data 连续段边界挑选下一轮 pilot。
    规则：
      - data_pos 为空：返回空数组（无需 pilot）
      - 对每个连续段 [L, R]：
          右侧：取位于 R 之后的连续 <= k 个可选 pilot
          左侧：取位于 L 之前的连续 <= k 个可选 pilot
      - 去重、升序，只从 available_pilots 中选，绝不越界
    """
    # 规范化输入
    data_pos = np.array(data_pos, dtype=int).reshape(-1)
    available_pilots = np.array(available_pilots, dtype=int).reshape(-1)
    if data_pos.size == 0:
        return np.zeros(0, dtype=int)
    if available_pilots.size == 0 or edge_expand_k <= 0:
        return np.zeros(0, dtype=int)

    data_pos = np.unique(data_pos)
    available_pilots = np.unique(available_pilots)
    P = available_pilots.size

    # ---- 自行分段：把 data_pos 拆成 [(L1,R1), (L2,R2), ...] ----
    segments = []
    start = prev = data_pos[0]
    for v in data_pos[1:]:
        if v == prev + 1:
            prev = v
        else:
            segments.append((start, prev))
            start = prev = v
    segments.append((start, prev))

    chosen = []
    for L, R in segments:
        # 右侧：第一个 >= R+1 的 pilot 起，取连续 k 个
        ridx = np.searchsorted(available_pilots, R + 1, side="left")
        if ridx < P:
            r_end = min(ridx + edge_expand_k, P)
            if r_end > ridx:
                chosen.extend(available_pilots[ridx:r_end].tolist())

        # 左侧：最后一个 <= L-1 的 pilot 起，向左取连续 k 个
        lidx = np.searchsorted(available_pilots, L, side="left") - 1
        if lidx >= 0:
            l_start = max(lidx - edge_expand_k + 1, 0)
            if lidx + 1 > l_start:
                chosen.extend(available_pilots[l_start:lidx + 1].tolist())

    if not chosen:
        return np.zeros(0, dtype=int)

    chosen = np.array(chosen, dtype=int)
    chosen = np.unique(chosen)  # 去重 + 升序
    # 保险：只保留可用集合中的
    mask = np.isin(chosen, available_pilots)
    return chosen[mask]
