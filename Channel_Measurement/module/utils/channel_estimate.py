# -*- coding: utf-8 -*-
"""
utils/channel_estimate.py
Channel estimation, drift fitting, segment building, and pilot analysis.

- evaluate_H_f / correct_H_f / estimate_drift_and_origin
- _fit_drift_between / _pilot_quality
- build_segments_from_pilots: build per-symbol channel references from preamble + comb pilots
- analyze_pilots: vectorized pilot quality analysis (SNR, BER, SER)
- estimate_M_from_filesize / choose_next_pilots
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional

from .math_process import phase_unwrap_auto, fitting_line, segment_means_on
from .demodulate import _qpsk_hard, QPSK_reflection, get_constellation
from .metric import snr_from_constellation, _mad_sigma, _esno_from_sigmas


# ========= H(f) estimation / extrapolation =========

def evaluate_H_f(symbols_td: np.ndarray,
                 pilots_fd: np.ndarray | None,
                 DATA_BINS: np.ndarray | None = None) -> np.ndarray:
    """
    Unified H(f) estimation: supports 1D or 2D time-domain symbols.
    - symbols_td: [N] or [ns, N] (time-domain, CP removed)
    - pilots_fd : same dimensions
    Returns aligned with symbols_td.
    """
    pilots_fd = np.where(pilots_fd == 0, np.nan, pilots_fd)
    X = np.asarray(symbols_td)
    assert pilots_fd is not None
    eps = 1e-6
    pilots_fd = np.where(pilots_fd == 0.0, eps, pilots_fd)
    if X.ndim == 1:
        Yf = np.fft.fft(X)
        if DATA_BINS is None:
            DATA_BINS = np.arange(X.size)
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
        if pilots_fd.shape[1] > DATA_BINS.size:
            pilots_fd = pilots_fd[:, DATA_BINS]
        Y_f_used = Yf[:, DATA_BINS]
        H_fs = np.full(X.shape, np.nan, dtype=np.complex128)
        H_fs[:, DATA_BINS] = Y_f_used / np.asarray(pilots_fd)
        return H_fs
    else:
        raise ValueError("symbols_td must be 1D or 2D")


def correct_H_f(origin_H_f: np.ndarray,
                N: int,
                index: Optional[int, np.ndarray],
                symbol_len: int,
                delta: Optional[float, np.ndarray],
                fixed_phase_shift_factor: Optional[float, np.ndarray] = 0.0) -> np.ndarray:
    """Apply linear phase rotation (SFO/CFO) and CPE correction to H(f)."""
    k = np.linspace(-N // 2, N // 2, N, endpoint=False, dtype=np.int32)
    k = np.concatenate([k[N // 2:], k[:N // 2]])
    if isinstance(index, np.ndarray) and index.size > 1:
        if origin_H_f.ndim > 1:
            k, _ = np.broadcast_arrays(k, origin_H_f)
            index, _ = np.broadcast_arrays(index.reshape(-1, 1), origin_H_f)
        else:
            index, origin_H_f = np.broadcast_arrays(index.reshape(-1, 1), origin_H_f.reshape(1, -1))
            k, _ = np.broadcast_arrays(k, origin_H_f)
        if isinstance(delta, np.ndarray):
            delta = delta.reshape(-1, 1)
        if isinstance(fixed_phase_shift_factor, np.ndarray):
            fixed_phase_shift_factor = fixed_phase_shift_factor.reshape(-1, 1)
    linear_phase = np.exp(-1j * 2 * np.pi / N * (delta * index * symbol_len) * k)
    cpe = np.exp(1j * fixed_phase_shift_factor * index)
    return origin_H_f * linear_phase * cpe


def estimate_drift_and_origin(Hf_seq: np.ndarray,
                              *, N: int, symbol_len: int,
                              DATA_BINS: np.ndarray | None = None,
                              return_plot_args: bool = False, mode='each'):
    """
    Fit (delta, phi_step) from [ns, N] H(f) sequence (preamble pilots),
    align all H to average and obtain origin_Hf. mode in ['each', 'total'].
    """
    if DATA_BINS is None:
        DATA_BINS = np.arange(N)
    H = np.asarray(Hf_seq)
    assert H.ndim == 2
    ratios = H[1::1] / H[:-1:1]
    xs, phases, slopes, intercepts, deltas, phis = [], [], [], [], [0], [0]
    for ratio in ratios:
        x_auto, auto_unwrapped_phase, _ = phase_unwrap_auto(data=ratio[DATA_BINS], DATA_BINS=DATA_BINS, N=N)
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
            delta=np.sum(deltas[:idx + 1]),
            N=N, index=-1, symbol_len=symbol_len,
            fixed_phase_shift_factor=np.sum(phis[:idx + 1])
        ))
    origin = np.array(origin)
    origin = np.average(origin, axis=0)

    if mode == 'total':
        ratio = np.mean(H[1:] / H[:-1], axis=0)[DATA_BINS]
        x_auto, auto_unwrapped_phase, _ = phase_unwrap_auto(data=ratio, DATA_BINS=DATA_BINS, N=N)
        slope, intercept = fitting_line(x=x_auto, y=auto_unwrapped_phase, filter=True, residual_th=1.2)
        plot_args = {
            'ratio': ratio, 'slope': np.array(slope), 'intercept': np.array(intercept),
            'x_auto': np.array(x_auto), 'auto_unwrapped_phase': np.array(auto_unwrapped_phase), 'N': N
        }
        if not return_plot_args:
            return np.sum(deltas).astype(float) / (num_pilot - 1), np.sum(phis).astype(float) / (num_pilot - 1), origin
        else:
            return np.sum(deltas).astype(float) / (num_pilot - 1), np.sum(phis).astype(float) / (num_pilot - 1), origin, plot_args
    elif mode == 'each':
        plot_args = {
            'ratio': ratios[:, DATA_BINS],
            'slope': np.array(slopes), 'intercept': np.array(intercepts),
            'x_auto': np.array(xs), 'auto_unwrapped_phase': np.array(phases), 'N': N
        }
        if not return_plot_args:
            return np.array(deltas).astype(float), np.array(phis).astype(float), origin
        else:
            return np.array(deltas).astype(float), np.array(phis).astype(float), origin, plot_args
    else:
        raise ValueError(f"unknown mode {mode}")


# ========= Segment building =========

def _fit_drift_between(H_start, H_end, gap, N, *,
                       symbol_len=None, return_phi=False,
                       plot: bool | int = False,
                       DATA_BINS: np.ndarray | None = None):
    """Fit per-OFDM delta and phi_step from ratio of two H estimates separated by gap."""
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
        plot_unwrap_phase_fitting(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N)
    elif type(plot) is int:
        from .plot import plot_unwrap_phase_fitting
        plot_unwrap_phase_fitting(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N, title=f'pilot {plot}')

    delta = slope / (gap * symbol_len * (-2 * np.pi) / N)
    phi_step = intercept / gap
    if return_phi:
        return float(delta), float(phi_step)
    return float(delta)


def _pilot_quality(symbol_td: np.ndarray,
                   H_ref_start: np.ndarray,
                   *,
                   delta: float, phi_step: float, gap: int,
                   pilot_ref_fd: np.ndarray,
                   DATA_BINS: np.ndarray, N: int,
                   symbol_len: int) -> tuple[float, float]:
    """Evaluate pilot quality by equalizing with predicted H from previous segment reference."""
    H_pred = correct_H_f(
        origin_H_f=H_ref_start, N=N, index=gap, symbol_len=symbol_len,
        delta=delta, fixed_phase_shift_factor=phi_step
    )
    Xd = get_constellation(symbols_td=symbol_td, H_used=H_pred, DATA_BINS=DATA_BINS)
    Rd = pilot_ref_fd[DATA_BINS] if pilot_ref_fd.size == N else pilot_ref_fd
    snr_sc = snr_from_constellation(Xd, Rd)
    snr_db_med = float(10.0 * np.log10(np.median(np.clip(snr_sc, 1e-12, None))))
    q = 1.0 / (1.0 + np.exp(-(snr_db_med - 5.0) / 2.0))
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
                               symbol_len: int = 0,
                               N: int = 0,
                               fs: float = 48000,
                               delta_interpolator=None,
                               delta_global: float = 0.0,
                               phi_global: float = 0.0,
                               symbols_comb_td: Optional[np.ndarray] = None,
                               pilot_ref_comb_fd: Optional[np.ndarray] = None,
                               distance_power: float = 1.0,
                               eps: float = 1.0) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[np.ndarray, float]]]:
    """
    Build segment-level channel references and drift parameters from preamble+comb pilots.
    Returns per-data-symbol H references and fusion weights (w1, w2).
    """
    all_idx = np.arange(M, dtype=int)
    pilot_pos = np.asarray(pilot_pos, dtype=int)
    data_pos = np.setdiff1d(all_idx, pilot_pos) if data_pos is None else data_pos
    n_data = data_pos.size

    segs = []
    H_s_list, d_list, p_list, g_list = [], [], [], []
    prev_idx = -1 if start_idx is None else start_idx
    H_ref = H_start.copy()

    if Hf_comb.size > 0 and pilot_pos.size > 0:
        for j, pidx in enumerate(pilot_pos):
            gap = int(pidx - prev_idx)
            d_j, p_j = _fit_drift_between(H_ref, Hf_comb[j], gap, N=N, symbol_len=symbol_len,
                                          return_phi=True, DATA_BINS=DATA_BINS, plot=False)
            delta_interpolator.update(prev_idx, pidx, d_j)
            H_s_list.append(H_ref.copy())
            d_list.append(d_j); p_list.append(p_j); g_list.append(gap)
            segs.append((prev_idx, int(pidx), H_ref.copy(), Hf_comb[j].copy(),
                         float(d_j), float(p_j), j, gap))
            prev_idx = int(pidx)
            H_ref = Hf_comb[j].copy()
        if prev_idx < M - 1:
            if segs:
                last_d, last_p = segs[-1][4], segs[-1][5]
            else:
                last_d, last_p = float(delta_global), float(phi_global)
            H_end = correct_H_f(origin_H_f=H_ref, N=N,
                                index=int(M - prev_idx), symbol_len=symbol_len,
                                delta=last_d, fixed_phase_shift_factor=last_p)
            segs.append((prev_idx, M, H_ref.copy(), H_end.copy(),
                         float(last_d), float(last_p), len(pilot_pos) - 1, int(M - prev_idx)))
    else:
        H_end = correct_H_f(origin_H_f=H_ref, N=N,
                            index=int(M - prev_idx), symbol_len=symbol_len,
                            delta=delta_global, fixed_phase_shift_factor=phi_global)
        segs.append((prev_idx, M, H_start.copy(), H_end.copy(),
                     float(delta_global), float(phi_global), -1, int(M - prev_idx)))

    # Quality per segment
    q_list = None
    if q_comb is not None:
        q_list = np.asarray(q_comb).astype(float)
    elif (symbols_comb_td is not None) and (pilot_ref_comb_fd is not None) and \
         (len(segs) > 0) and (pilot_pos.size > 0):
        q_vals = []
        for (start_idx, end_idx, Hs, Hn, d_j, p_j, j_idx, gap) in segs:
            if j_idx < 0 or j_idx >= pilot_pos.size:
                q_vals.append(q_vals[-1] if q_vals else 1.0)
                continue
            q_j, _ = _pilot_quality(
                symbol_td=symbols_comb_td[j_idx], H_ref_start=Hs,
                delta=d_j, phi_step=p_j, gap=gap,
                pilot_ref_fd=pilot_ref_comb_fd[j_idx],
                DATA_BINS=DATA_BINS, N=N, symbol_len=symbol_len
            )
            q_vals.append(float(q_j))
        q_list = np.asarray(q_vals, dtype=float)

    # Per-symbol mapping
    H_start_per = np.empty((n_data, N), dtype=H_start.dtype)
    H_near_per = np.empty((n_data, N), dtype=H_start.dtype)
    delta_from_s = np.empty(n_data, dtype=float)
    delta_from_e = np.empty(n_data, dtype=float)
    phi_per = np.empty(n_data, dtype=float)
    dt1 = np.empty(n_data, dtype=float)
    dt2 = np.empty(n_data, dtype=float)
    w1 = np.empty(n_data, dtype=float)
    w2 = np.empty(n_data, dtype=float)

    ptr = 0
    for (start_idx, end_idx, Hs, Hn, d_j, p_j, j_idx, gap) in segs:
        mask = (data_pos > start_idx) & (data_pos <= end_idx)
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        i_vals = data_pos[idxs].astype(int)
        dt_from_start = i_vals - float(start_idx)
        near_idx = float(end_idx)
        dt_from_near = i_vals - near_idx
        d1 = np.maximum(dt_from_start, 0.0)
        d2 = np.abs(dt_from_near)
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
        H_start_per[ptr:ptr + cnt, :] = Hs[None, :]
        H_near_per[ptr:ptr + cnt, :] = Hn[None, :]
        delta_from_s[ptr:ptr + cnt] = delta_interpolator.get_interp_delta(i_vals, start_idx)
        delta_from_e[ptr:ptr + cnt] = delta_interpolator.get_interp_delta(i_vals, end_idx if end_idx < M else start_idx)
        phi_per[ptr:ptr + cnt] = p_j
        dt1[ptr:ptr + cnt] = dt_from_start
        dt2[ptr:ptr + cnt] = dt_from_near
        w1[ptr:ptr + cnt] = w1_seg
        w2[ptr:ptr + cnt] = w2_seg
        ptr += cnt

    if ptr < n_data:
        raise RuntimeError("Unexpected Error! check the logic in segments building")

    pilot_seg = {'H_start': np.array(H_s_list), 'delta': np.array(d_list),
                 'phi': np.array(p_list), 'gap': np.array(g_list)}
    data_seg = {
        "H_start_per_seg": H_start_per, "delta_per_seg": delta_from_s,
        "phi_per_seg": phi_per, "dt_from_start_per_sym": dt1,
        "H_near_per_sym": H_near_per, "delta_per_seg_per_sym": delta_from_e,
        "phi_per_seg_per_sym": phi_per, "dt_from_near_per_sym": dt2,
        "w1_per_sym": w1, "w2_per_sym": w2,
    }
    return pilot_seg, data_seg


# ========= Pilot analysis =========

def analyze_pilots(symbols_td: np.ndarray | None,
                   Hf: np.ndarray | None,
                   pilot_ref: np.ndarray,
                   DATA_BINS: np.ndarray,
                   *,
                   mode: str = "front",
                   clockwise: bool = False,
                   symbols_fd: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """Vectorized pilot quality analysis: SNR, Es/N0, sigma, BER, SER under non-coincident equalization."""
    pilot_ref = np.asarray(pilot_ref)
    if pilot_ref.ndim == 1:
        pilot_ref = pilot_ref[None, :]
    Rd = pilot_ref[:, DATA_BINS] if pilot_ref.shape[1] > DATA_BINS.size else pilot_ref
    if symbols_fd is None:
        assert symbols_td is not None and Hf is not None
        symbols_td = np.asarray(symbols_td)
        Hf = np.asarray(Hf)
        Xd = get_constellation(symbols_td=symbols_td, H_used=Hf, DATA_BINS=DATA_BINS)
    else:
        if symbols_fd.ndim == 1:
            symbols_fd = symbols_fd[None, :]
        Xd = symbols_fd[:, DATA_BINS] if symbols_fd.shape[1] > DATA_BINS.size else symbols_fd
    assert Xd.shape == Rd.shape
    snr_sc = snr_from_constellation(Xd, Rd)
    snr_db_med = 10.0 * np.log10(np.median(np.clip(snr_sc, 1e-12, None), axis=-1))
    quality = 1.0 / (1.0 + np.exp(-(snr_db_med - 5.0) / 2.0))
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
        "snr_db_med": snr_db_med.astype(float), "esno_db": esno_db.astype(float),
        "q": quality.astype(float), "sigma_r": sigma_r.astype(float),
        "sigma_i": sigma_i.astype(float), "ber": ber.astype(float), "ser": ser.astype(float)
    }


# ========= M estimation & pilot selection =========

def equalize_first_data_symbol(symbols_all_td: np.ndarray,
                               origin_Hf: np.ndarray, delta: float, phi_step: float,
                               *, N: int, DATA_BINS: np.ndarray,
                               symbol_len: int | None = None) -> np.ndarray:
    """Equalize the first data OFDM symbol using corrected H(f)."""
    if symbol_len is None:
        symbol_len = N
    H1 = correct_H_f(origin_H_f=origin_Hf, delta=delta, N=N, symbol_len=symbol_len,
                     index=1, fixed_phase_shift_factor=phi_step)
    return get_constellation(symbols_td=symbols_all_td[0], H_used=H1, DATA_BINS=DATA_BINS)


def estimate_M_from_filesize(*, filesize_bytes: int, K: int, Ncw: int, Nd: int,
                             modulation_bits: int, interval: int) -> int:
    """Estimate total OFDM symbol count from file size and code parameters."""
    info_bits = int(filesize_bytes) * 8
    n_codewords = (info_bits + K - 1) // K
    coded_bits = n_codewords * Ncw
    bits_per_ofdm = Nd * modulation_bits
    data_syms = (coded_bits + bits_per_ofdm - 1) // bits_per_ofdm
    comb_syms = np.ceil(data_syms / interval) - 1 if interval else 0
    return int(data_syms + comb_syms)


def choose_next_pilots(data_pos: np.ndarray,
                       available_pilots: np.ndarray,
                       edge_expand_k: int = 1) -> np.ndarray:
    """Select next pilot indices by expanding around data segment boundaries."""
    data_pos = np.array(data_pos, dtype=int).reshape(-1)
    available_pilots = np.array(available_pilots, dtype=int).reshape(-1)
    if data_pos.size == 0:
        return np.zeros(0, dtype=int)
    if available_pilots.size == 0 or edge_expand_k <= 0:
        return np.zeros(0, dtype=int)
    data_pos = np.unique(data_pos)
    available_pilots = np.unique(available_pilots)
    P = available_pilots.size
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
        ridx = np.searchsorted(available_pilots, R + 1, side="left")
        if ridx < P:
            r_end = min(ridx + edge_expand_k, P)
            if r_end > ridx:
                chosen.extend(available_pilots[ridx:r_end].tolist())
        lidx = np.searchsorted(available_pilots, L, side="left") - 1
        if lidx >= 0:
            l_start = max(lidx - edge_expand_k + 1, 0)
            if lidx + 1 > l_start:
                chosen.extend(available_pilots[l_start:lidx + 1].tolist())
    if not chosen:
        return np.zeros(0, dtype=int)
    chosen = np.array(chosen, dtype=int)
    chosen = np.unique(chosen)
    mask = np.isin(chosen, available_pilots)
    return chosen[mask]
