# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import argparse
import time

from ..utils.print_aid import print_padded, print_dict_values
from ..utils.demodulate import get_symbols, get_constellation
from ..utils.encode import ldpc_encode_bits, ldpc_make_code, scramble_bits
from ..utils.modulate import generate_chirp, serial_to_parallel, QPSK_mapping
from ..utils.synchronize import synchronize
from ..utils.decode import ldpc_decode_blocks
from ..utils.io_interface import get_bits_from_file, get_bits_from_str, num_to_bits_msb
from ..utils.plot import (plot_correlation, plot_received_signal, plot_original_constellations,
                          plot_corrected_constellations, plot_data_constellations, plot_unwrap_phase_fitting,
                          plot_impulse_response, plot_snr_over_time, plot_snr_over_subcarrier, plot_pre_post_ber)
from ..utils.DeltaInterpolator import DeltaInterpolator

# 待归类函数
from ..utils.batch import (
    # Pilot/Comb 处理
    evaluate_H_f, correct_H_f,
    estimate_drift_and_origin,
    analyze_pilots,
    build_segments_from_pilots,
    generate_comb_pilot_symbol,
    # 矢量化均衡/PLL/噪声/收缩/LLR
     pll_snr_median, snr_from_constellation,
    robust_sigma, mmse_shrinkage, llr_from_constellation,
    llr_scale_by_snr, pack_llr_blocks, apply_cpe_pll_sequence,
    # M 估计
    estimate_M_from_filesize,
    # next turn pilot
    choose_next_pilots
)


def _build_ofdm_maps(ofdm_idx_blocks: np.ndarray) -> tuple[dict, dict, int]:
    """
    输入: ofdm_idx_blocks [B, Ncw]
    输出:
      - ofdm_to_blocks: {ofdm -> np.array(block_ids)}
      - ofdm_to_flat_idx: {ofdm -> np.array(flat positions 按原flatten顺序，长度应为 Nd*2)}
      - num_blocks: B
    """
    B, Ncw = ofdm_idx_blocks.shape
    flat_ofdm = ofdm_idx_blocks.reshape(-1).astype(np.int64)
    ofdm_to_blocks = {}
    for b in range(B):
        for o in np.unique(ofdm_idx_blocks[b]):
            ofdm_to_blocks.setdefault(int(o), set()).add(b)
    ofdm_to_blocks = {k: np.array(sorted(list(v)), dtype=np.int32) for k, v in ofdm_to_blocks.items()}
    ofdm_to_flat_idx = {}
    # 直接使用 np.where(flat_ofdm==o) 的索引顺序，即为原行主序(OFDM优先)的顺序
    uniq = np.unique(flat_ofdm)
    for o in uniq:
        idx = np.nonzero(flat_ofdm == o)[0]
        ofdm_to_flat_idx[int(o)] = idx
    return ofdm_to_blocks, ofdm_to_flat_idx, B


def receiver(rx: np.ndarray, pilot: np.ndarray, args: argparse.Namespace):
    """
    入口：
        rx:    1D 复数数组（整段时域，含 chirp + pilot + data/comb）
        pilot: [num_pilot, N] 的频域参考导频
        args:  argparse.Namespace，字段参见下方读取
    返回：
        decoded_bits_scr: 最终解码后再加扰的比特流（含 64bit 头）
        info:              统计信息字典
    """
    # ---------------- 基本参数 ----------------
    fs              = args.fs
    N               = args.N
    cp_len          = args.cp_len
    num_pilot       = args.num_pilot
    chirp_len       = args.chirp_len
    chirp_l         = args.chirp_l
    chirp_h         = args.chirp_h
    clockwise       = args.clockwise

    # comb pilot settings
    INTERVAL        = args.INTERVAL
    comb_seed_base  = args.COMB_PILOT_SEED_BASE

    # groundtruth settings
    groundtruth     = args.groundtruth
    tx_file_path    = getattr(args, "tx_file_path", None)

    # other settings
    head_bit        = args.head_bit

    # PLOT
    plot            = args.plot
    plot_opt        = args.plot_opt

    # PRINT
    print_flag      = args.print_flag
    print_opt       = args.print_opt
    print_len       = getattr(args, 'print_len', 64)
    print_pad       = getattr(args, 'print_pad', '-')

    symbol_len = N + cp_len
    POS_BINS   = np.arange(1, N//2)
    DATA_BINS  = POS_BINS[args.data_start:-args.data_tail]
    Nd         = DATA_BINS.size

    code = ldpc_make_code(
        standard=args.ldpc_standard, rate=args.ldpc_rate, z=args.ldpc_z, ptype=args.ldpc_ptype,
        device=args.ldpc_device, llr_clip=args.ldpc_llr_clip, max_iter=args.ldpc_max_iter,
        verbose=args.ldpc_verbose, log_every=args.ldpc_log_every, check_every=args.ldpc_check_every,
        microbatch=args.ldpc_microbatch, print_iter=args.ldpc_print_iter
    )

    delta_interpolator = DeltaInterpolator(fs=fs, method=args.interp_mode, smooth=args.interp_smooth)

    # ---------------- Groundtruth数据准备 ----------------
    if groundtruth:
        assert tx_file_path is not None, "groundtruth=True 需要提供 --tx_file_path"
        print(f"use groundtruth, tx_file_path: {tx_file_path}")
        gt_bits_raw = get_bits_from_file(tx_file_path)
        if args.type_bit_w:
            tail = tx_file_path.split('.')[-1]
            type_bit = get_bits_from_str(args.suffix_map[tail])
            assert int(type_bit.size) == args.type_bit_w, f"type suffix cannot use {args.type_bit_w} bits to express"
        else:
            type_bit = np.array([])
        size_bit = num_to_bits_msb(int(gt_bits_raw.size), bit_num=args.size_bit_w)
        if head_bit:
            gt_bits_raw = np.concatenate([type_bit, size_bit, gt_bits_raw])
        if args.use_scrambler:
            gt_bits_scr = scramble_bits(gt_bits_raw,
                                        seed=args.scrambler_seed,
                                        mode=args.scrambler_mode,
                                        bit_width=args.scrambler_bitwidth)
            gt_bits_ldpc, _ = ldpc_encode_bits(gt_bits_scr, c=code)
        else:
            gt_bits_ldpc, _ = ldpc_encode_bits(gt_bits_raw, c=code)
        const_data_global_ref = QPSK_mapping(serial_to_parallel(gt_bits_ldpc, N=(Nd + 1) * 2))

    # ---------------- 0) 时域同步（scipy.signal.correlate） ----------------
    if print_flag: print_padded("begin to synchronize", print_len, print_pad)
    chirp_tpl = generate_chirp(fs, duration=chirp_len, f_l=chirp_l, f_h=chirp_h)
    ofdm_start, corr, chirp_start = synchronize(rx, chirp_tpl, mode="full")
    if plot and plot_opt['correlation']:
        plot_correlation(corr=corr,
                         axvline_dict={'chirp_start':chirp_start+chirp_tpl.size-1,
                                       'ofdm_start':ofdm_start+chirp_tpl.size-1})
    if plot and plot_opt['received_signal']:
        plot_received_signal(rx, ofdm_start, num_pilot, N, cp_len)
    if print_flag: print_padded("synchronize done", print_len, print_pad)

    # ---------------- 1) 前导 pilot：H(f) 与漂移/基准估计 + 质量评估 ----------------
    if print_flag: print_padded("begin to analyze front pilot", print_len, print_pad)
    rx_pilot_td = rx[ofdm_start : ofdm_start + num_pilot * (N + cp_len)]
    sym_pilot_td = get_symbols(rx_pilot_td, cp_len=cp_len, N=N)                      # [num_pilot, N] 时域
    Hf_pilot = evaluate_H_f(sym_pilot_td, pilots_fd=pilot)      # [num_pilot, N]
    res_arg = estimate_drift_and_origin(Hf_pilot, N=N, symbol_len=symbol_len, DATA_BINS=DATA_BINS,
                                        return_plot_args=plot_opt['unwrap'], mode='total')
    delta0 = np.median(res_arg[0]) if res_arg[0].size > 1 else res_arg[0]
    phi0 = np.median(res_arg[1]) if res_arg[1].size > 1 else res_arg[1]
    origin_H_f = res_arg[2]
    freq_bias = fs / (delta0 + 1) - fs

    if isinstance(res_arg[0], np.ndarray) and res_arg[0].size > 1:
        delta_interpolator.update(
            idx_s=np.linspace(-num_pilot, -2, num_pilot-1).astype(int),
            idx_e=np.linspace(-num_pilot+1, -1, num_pilot-1).astype(int),
            delta=res_arg[0][1:]
        )
        pilot_metrics = analyze_pilots(
            symbols_td=sym_pilot_td, pilot_ref=pilot, DATA_BINS=DATA_BINS, mode="front", clockwise=clockwise,
            Hf=correct_H_f(origin_H_f=origin_H_f, N=N, index=np.arange(num_pilot), symbol_len=symbol_len,
                           delta=np.concatenate([np.zeros(1), np.array([np.sum(res_arg[0][1:i+1])/i for i in range(1,num_pilot)])], axis=0),
                           fixed_phase_shift_factor=np.concatenate([np.zeros(1), np.array([np.sum(res_arg[1][1:i+1])/i for i in range(1,num_pilot)])], axis=0))
        )
    else:
        delta_interpolator.update(-num_pilot,-1,res_arg[0])
        pilot_metrics = analyze_pilots(
            symbols_td=sym_pilot_td, pilot_ref=pilot, DATA_BINS=DATA_BINS, mode="front", clockwise=clockwise,
            Hf=correct_H_f(origin_H_f=origin_H_f, N=N, index=np.arange(num_pilot), symbol_len=symbol_len,
                           delta=delta0, fixed_phase_shift_factor=phi0)
        )

    if print_flag:
        print(f"delta:{delta0}")
        print(f"fixed_phase_shift_factor:{phi0}")
        print(f"fs of receiver - fs of emitter = {freq_bias}")
        print(f"front pilot metrics: \nindex    snr     ber")
        if print_opt['pilot_metric']:
            print_dict_values(pilot_metrics, ["snr_db_med","ber"], [f"pilot {i+1}" for i in range(num_pilot)])

    if plot and plot_opt['impulse_response']:
        plot_impulse_response(H_f=origin_H_f, fs=fs)
    if plot and plot_opt['unwrap']:
        plot_args = res_arg[3]
        plot_unwrap_phase_fitting(
            phase_shift=plot_args['ratio'],
            slope=plot_args['slope'],
            intercept=plot_args['intercept'],
            x_auto=plot_args['x_auto'],
            auto_unwrapped_phase=plot_args['auto_unwrapped_phase'],
            N=plot_args['N']
        )
    if plot and plot_opt['raw_pilot_constellation']:
        plot_original_constellations(symbols_td=sym_pilot_td, H_used=Hf_pilot[0], pilot=pilot, DATA_BINS=DATA_BINS)
    if plot and plot_opt['corrected_pilot_constellation']:
        plot_corrected_constellations(symbols_td=sym_pilot_td, origin_H_f=origin_H_f, pilot=pilot,
                                      symbol_len=symbol_len, delta=res_arg[0], fixed_phase_shift_factor=res_arg[1],
                                      DATA_BINS=DATA_BINS)
    # from ..utils.plot import plot_decoding_process_demo
    # plot_decoding_process_demo(symbols_td=sym_pilot_td, origin_H_f=origin_H_f, symbols_fd=pilot, DATA_BINS=DATA_BINS,
    #                            symbol_len=symbol_len, delta=res_arg[0], fixed_phase_shift_factor=res_arg[1])
    #
    # return np.ones(1), dict()
    if plot and plot_opt.get('snr_time_pilot', False):
        plot_snr_over_time(pilot_metrics["snr_db_med"], title="Front Pilot SNR over OFDM symbols")
    if print_flag: print_padded(f"front pilot analysis done", print_len, print_pad)

    # ---------------- 2) 数据段切片 ----------------
    rx_data_td = rx[ofdm_start + num_pilot * (N + cp_len):]
    symbols_all_td = get_symbols(rx_data_td, N=N, cp_len=cp_len)        # [M_guess, N]
    M_try = min(int(symbols_all_td.shape[0] * getattr(args, 'first_try_portion', 0.8)),
                estimate_M_from_filesize(filesize_bytes=code.K * args.ldpc_batch // 8,
                                         K=code.K, Ncw=code.N, Nd=Nd,
                                         modulation_bits=2,interval=INTERVAL)
                )

    # ---------------- 3) First Try Parameter----------------
    head_decoded_done = False
    blk_num_head = np.ceil(head_bit/code.K).astype(int)
    M = M_try
    all_idx = np.arange(M_try)
    pilot_pos_global = all_idx[(all_idx % (INTERVAL + 1)) == INTERVAL] if INTERVAL is not None else np.zeros(0)
    data_pos_global = all_idx[(all_idx % (INTERVAL + 1)) != INTERVAL] if INTERVAL is not None else all_idx
    delta_interpolator.set_params(node_pos=np.linspace(-num_pilot, M-1, M+num_pilot).astype(int))
    if print_flag: print_padded(f"first try to solve {M_try} OFDM symbols", print_len, print_pad)


    max_pseudo_iter = getattr(args, "pseudo_pilot_max_iter", 20)
    alpha_pseudo_H = getattr(args, "pseudo_pilot_alpha", 0.5)  # H 融合平滑系数 [0,1]

    iter_pseudo = 0
    circle_flag = True
    pilot_pos = pilot_pos_global.copy() if args.use_comb else np.array([])
    data_pos = data_pos_global.copy()
    # ===== 全局 ofdm/频率索引 与 LLR 缓冲（行=OFDM in data_pos_global, 列=Nd*2） =====
    ofdm_full = symbols_all_td.shape[0]
    ofdm_idx_full = np.arange(ofdm_full)
    data_full = ofdm_idx_full[(ofdm_idx_full % (INTERVAL + 1)) != INTERVAL] if INTERVAL is not None else ofdm_idx_full
    freq_axis_full = np.linspace(0.0, fs, N, endpoint=False)
    ofdm_idx_global = np.repeat(data_full[:, None], Nd * 2, axis=1)
    sc_freq_global = np.repeat(np.repeat(freq_axis_full[DATA_BINS], 2)[None, :], data_full.shape[0], axis=0)
    llr_global = np.full((data_full.shape[0], Nd * 2), np.nan, dtype=np.float32)
    # 用“占位 LLR”pack一次，得到稳定的块划分与 ofdm→块 的映射
    _packed0 = pack_llr_blocks(ofdm_idx_global, sc_freq_global, np.nan_to_num(llr_global, nan=0.0), Ncw=code.N)
    ofdm2blk, ofdm2flat, num_blocks = _build_ofdm_maps(_packed0['ofdm_idx'])
    block_done = np.zeros(num_blocks, dtype=bool)  # 已通过 LDPC 的块  在解出head后需要截断
    info_blocks = np.full((num_blocks, code.K), -1, dtype=np.int8)  # 每块信息比特缓存（-1=未知） 在解出head后需要截断
    pos2ref = {}

    # comb 参考(按 DATA_BINS 截断为 Nd 列，以适配 evaluate_H_f(..., DATA_BINS))
    pilot_ref_comb_fd_global = np.stack(
        [generate_comb_pilot_symbol(N, comb_seed_base + i) for i in range(pilot_pos_global.size)], axis=0
    ).reshape(-1,N)[:, DATA_BINS] if pilot_pos_global.size and args.use_comb else np.zeros((0, Nd), complex)
    pilot_ref_comb_fd = pilot_ref_comb_fd_global.copy()

    # ---------------- 4) Pseudo pilot solving----------------
    start = time.time()
    while circle_flag and (iter_pseudo < max_pseudo_iter):
        n_comb = pilot_pos.size
        symbols_comb_td = (symbols_all_td[pilot_pos] if n_comb else np.zeros((0, N), complex))
        Hf_comb = evaluate_H_f(symbols_comb_td, pilot_ref_comb_fd, DATA_BINS) if n_comb else np.zeros((0, N), complex)

        # 段构建
        H_start = (Hf_pilot[-1] / np.max(np.abs(np.nan_to_num(Hf_pilot, nan=0.0)))
                   * np.max(np.abs(np.nan_to_num(Hf_comb.ravel(), nan=0.0)))) if n_comb else Hf_pilot[-1]
        pilot_pred_param, seg = build_segments_from_pilots(
            H_start=H_start, Hf_comb=Hf_comb, pilot_pos=pilot_pos, data_pos=data_pos,
            M=M, DATA_BINS=DATA_BINS, q_comb=None, mode="quality_distance",
            symbol_len=symbol_len, N=N, fs=fs, delta_global=delta0, phi_global=phi0,
            symbols_comb_td=symbols_comb_td, pilot_ref_comb_fd=pilot_ref_comb_fd,
            delta_interpolator=delta_interpolator
        )

        comb_metrics = (analyze_pilots(
            symbols_td=symbols_comb_td, pilot_ref=pilot_ref_comb_fd,
            DATA_BINS=DATA_BINS, mode="comb", clockwise=clockwise,
            Hf=correct_H_f(origin_H_f=pilot_pred_param['H_start'], delta=pilot_pred_param['delta'],
                           fixed_phase_shift_factor=pilot_pred_param['phi'], index=pilot_pred_param['gap'],
                           N=N, symbol_len=symbol_len)
            ) if n_comb else {"snr_db_med": np.array([]), "ber": np.array([])})


        if plot and plot_opt['snr_time_comb'] and n_comb:
            plot_snr_over_time(comb_metrics["snr_db_med"], title="Comb Pilot SNR over OFDM symbols", pos=pilot_pos)
        if plot and plot_opt['freq_offset_interpolate']:
            delta_interpolator.plot()
        if print_flag and n_comb:
            if print_opt['pilot_delta']:
                print('index    delta           phi')
                print_dict_values(pilot_pred_param, ['delta', 'phi'],[f"{i+1}: pilot {index+1}" for i, index in enumerate(pilot_pos)])
            if print_opt['pilot_metric']:
                print('index      snr         ber')
                print_dict_values(comb_metrics, ['snr_db_med', 'ber'],[f"{i+1}: pilot {index+1}" for i, index in enumerate(pilot_pos)])

        # 外推 H_used（两路 + 权重融合）
        H_used_from_start = correct_H_f(
            origin_H_f=seg["H_start_per_seg"], index=seg["dt_from_start_per_sym"], N=N, symbol_len=symbol_len,
            delta=seg["delta_per_seg"], fixed_phase_shift_factor=seg["phi_per_seg"]
        ).reshape(-1, N)
        H_used_from_near = correct_H_f(
            origin_H_f=seg["H_near_per_sym"], index=seg["dt_from_near_per_sym"], N=N, symbol_len=symbol_len,
            delta=seg["delta_per_seg_per_sym"], fixed_phase_shift_factor=seg["phi_per_seg_per_sym"]
        ).reshape(-1, N)

        w1, w2 = seg["w1_per_sym"], seg["w2_per_sym"]
        H_used = (H_used_from_start * w1[:, None] + H_used_from_near * w2[:, None]) / (w1[:, None] + w2[:, None] + 1e-12)

        sym_data_td = symbols_all_td[data_pos].reshape(-1, N)
        const_zf = get_constellation(sym_data_td, H_used, DATA_BINS=DATA_BINS)
        pll_snr_med = pll_snr_median(const_zf)
        const_pll = apply_cpe_pll_sequence(
            const_zf, pll_snr_med,
            alpha=getattr(args, "pll_alpha", 0.15),
            snr_th_db=getattr(args, "pll_snr_th_db", 6.0),
            alpha_min=getattr(args, "pll_alpha_min", 0.05),
            alpha_max=getattr(args, "pll_alpha_max", 0.50),
            snr_th_min_db=getattr(args, "pll_snr_min_db", 3.0),
            snr_th_max_db=getattr(args, "pll_snr_max_db", 10.0),
            beta=getattr(args, "pll_beta", 0.9),
            snr_mid_db=getattr(args, "pll_snr_mid_db", 6.0),
            snr_scale=getattr(args, "pll_snr_scale", 4.0),
        )
        # const_pll = const_zf
        # 噪声/收缩
        sigmas = robust_sigma(const_pll, per_sc=args.sig_trk_per_sc)
        Habs2 = np.abs(H_used[:, DATA_BINS])**2
        const_mmse = mmse_shrinkage(const_pll, Habs2, sigmas)
        # const_mmse = const_pll
        if groundtruth:
            const_ref = const_data_global_ref[np.where(np.isin(data_pos_global, data_pos))[0]]
            data_metrics = analyze_pilots(pilot_ref=const_ref, symbols_fd=const_mmse,DATA_BINS=np.arange(Nd),
                                          symbols_td=None, Hf=None,clockwise=clockwise, mode='data')
            if print_flag and print_opt['data_metric']:
                print('index      snr         ber')
                print_dict_values(data_metrics, ['snr_db_med', 'ber'], [f"{i+1}: data {index+1}" for i, index in enumerate(data_pos)])
            if plot and plot_opt['snr_time_data']:
                plot_snr_over_time(data_metrics["snr_db_med"], title="Data symbol SNR over OFDM symbols", pos=data_pos)
            if plot and plot_opt['data_constellation']:
                plot_data_constellations(const=const_mmse, data_pos=data_pos, const_ref=const_ref)
            if plot and plot_opt['snr_over_sc']:
                plot_snr_over_subcarrier(np.mean(snr_from_constellation(const_mmse, const_ref), axis=0),
                                         sc_idx=DATA_BINS, title="Average SNR over subcarriers (Data)")


        # LLR + 按 per-SC SNR(dB) 缩放
        llr_raw, stat = llr_from_constellation(const_mmse, llr_clip=args.ldpc_llr_clip)
        scale_sc = llr_scale_by_snr(stat["snr_db_per_sc"], lo=2.0, hi=10.0, min_scale=0.4, max_scale=1.0)
        llr_scaled = (llr_raw.reshape(-1, Nd, 2) * scale_sc[:, :, None]).reshape(-1, Nd*2)

        # === 3) 写入“全局 LLR 缓冲”，只覆盖本轮 data_pos 对应行 ===
        rows_global = np.nonzero(np.isin(data_pos_global, data_pos))[0]
        llr_global[rows_global, :] = llr_scaled

        # === 4) 基于固定块边界重新打包，并仅解码“就绪且未解”的块 ===
        packed_now = pack_llr_blocks(
            ofdm_idx=ofdm_idx_global,
            sub_carr_freq=sc_freq_global,
            llr=np.nan_to_num(llr_global, nan=0.0),
            Ncw=code.N
        )
        llr_blocks_all = packed_now['llr']  # [B, Ncw]
        ofdm_idx_blocks = packed_now['ofdm_idx']  # [B, Ncw]
        sc_freq_blocks = packed_now['sc_freq']  # [B, Ncw]

        # 计算每块“就绪”掩码：块内 Ncw 个比特均已填充（非 NaN）
        mask_flat = np.isfinite(llr_global.reshape(-1))
        num_blocks = llr_blocks_all.shape[0]
        block_ready = np.empty(num_blocks, dtype=bool)
        for b in range(num_blocks):
            s = b * code.N
            e = s + code.N
            block_ready[b] = np.all(mask_flat[s:e])

        to_decode = np.nonzero((~block_done) & block_ready)[0]
        if to_decode.size:
            llr_blocks = {
                'llr': llr_blocks_all[to_decode],
                'ofdm_idx': ofdm_idx_blocks[to_decode],
                'sc_freq': sc_freq_blocks[to_decode],
            }
            decoded_info_part, it, syn = ldpc_decode_blocks(
                llr_blocks=llr_blocks,
                code=code,
                batch=args.ldpc_batch,
                device=args.ldpc_device
            )
            decoded_info_part = decoded_info_part.reshape(-1, code.K)
            syn = syn.flatten()
            ok = (syn == 0)
            block_done[to_decode[ok]] = True
            info_blocks[to_decode] = decoded_info_part
        else:
            syn = np.array([], dtype=int)

        if np.all(block_done[:blk_num_head]) and not head_decoded_done:
            # head 所需的blk全部通过校验，head decoded done
            head_decoded_done = True
            if args.use_scrambler:
                decoded_head_raw = scramble_bits(info_blocks[:blk_num_head].ravel(),
                                                 seed=args.scrambler_seed,
                                                 mode=args.scrambler_mode,
                                                 bit_width=args.scrambler_bitwidth)
            else:
                decoded_head_raw = info_blocks[:blk_num_head].ravel()

            file_bits = 0
            size_bit = decoded_head_raw[args.type_bit_w: args.head_bit]
            for b in size_bit:
                file_bits = (file_bits << 1) | int(b)
            file_bytes = int(np.ceil(file_bits / 8.0))
            M = estimate_M_from_filesize(filesize_bytes=file_bytes, K=code.K, Ncw=code.N, Nd=Nd,
                                         modulation_bits=2, interval=INTERVAL)
            if print_flag: print_padded(f"decoded file head, {M} effective OFDM symbols", print_len, print_pad)
            # 用读取的文件头修正全局量
            all_idx = np.arange(M)
            pilot_pos_global = all_idx[(all_idx % (INTERVAL + 1)) == INTERVAL] if INTERVAL is not None else np.zeros(0)
            data_pos_global = all_idx[(all_idx % (INTERVAL + 1)) != INTERVAL] if INTERVAL is not None else all_idx

            delta_interpolator.set_params(node_pos=np.linspace(-num_pilot, M-1, M+num_pilot).astype(int))

            ofdm_idx_global = ofdm_idx_global[:data_pos_global.shape[0], :]
            sc_freq_global = sc_freq_global[:data_pos_global.shape[0],:]
            llr_global = llr_global[:data_pos_global.shape[0],:]
            _packed0 = pack_llr_blocks(ofdm_idx_global, sc_freq_global, np.nan_to_num(llr_global, nan=0.0), Ncw=code.N)
            ofdm2blk, ofdm2flat, num_blocks = _build_ofdm_maps(_packed0['ofdm_idx'])
            block_done = block_done[: num_blocks]
            info_blocks = info_blocks[: num_blocks]
            data_pos = data_pos_global.copy()
            pilot_pos = pilot_pos_global.copy() if args.use_comb else np.array([])
            pilot_ref_comb_fd_global = np.stack(
                [generate_comb_pilot_symbol(N, comb_seed_base + i) for i in range(pilot_pos_global.size)], axis=0
            ).reshape(-1, N)[:, DATA_BINS] if pilot_pos_global.size and args.use_comb else np.zeros((0, Nd), complex)


        if print_flag or print_opt['iter_verbose']:
            nz = int(np.count_nonzero(syn != 0)) if syn.size else 0
            print_padded(f"[iter {iter_pseudo}] process_blocks={to_decode.size}, new_blocks={to_decode.size-nz},"
                         f" syn_nonzero={nz}, done={int(np.count_nonzero(block_done))}/{num_blocks}",
                         print_len, print_pad)

        # === 5) 选择“可晋升”的 data OFDM：覆盖它的全部块均已通过 ===
        promotable = []
        for o in data_pos.tolist():
            blks = ofdm2blk.get(int(o), None)
            idxs = ofdm2flat.get(int(o), np.array([], dtype=int))
            if blks is None or blks.size == 0:
                continue
            # 需要该 OFDM 的 Nd*2 位都在打包范围内（尾部不足时跳过）
            if (idxs.size >= Nd * 2) and np.all(block_done[blks]):
                promotable.append(int(o))

        # === 6) 仅在“最后 update”处更新 data_pos / pilot_pos / pilot_ref_comb_fd ===
        # 下一轮 data = 仍有未完成块的 ofdm
        still_data = []
        for o in data_pos.tolist():
            blks = ofdm2blk.get(int(o), None)
            if blks is None or blks.size == 0:
                continue
            if not np.all(block_done[blks]):
                still_data.append(int(o))
        data_pos = np.array(still_data, dtype=int)

        # 若既无新块可解码又无可晋升 OFDM，则终止
        if (len(promotable) == 0) or (data_pos.size == 0):
            if data_pos.size > 10:
                print(f"Unsolved data OFDM symbols: {data_pos.size}: {data_pos.ravel()[:10]} ...")
            else:
                print(f"Unsolved data OFDM symbols: {data_pos.size}: {data_pos.ravel()}")
            break

        # 下一轮 pilot
        pilot_pos = choose_next_pilots(data_pos=data_pos, edge_expand_k=getattr(args, "edge_expand", 2),
                                       available_pilots=np.array(promotable))
        # np.union1d(np.array(promotable), pilot_pos)           np.array(promotable)

        # 1) 原 comb 导频参考（Nd 列）
        if pilot_pos_global.size:
            for i, p in enumerate(pilot_pos_global):
                if p in set(pilot_pos.tolist()):
                    pos2ref[int(p)] = pilot_ref_comb_fd_global[i]  # Nd 维

        # 2) 伪导频参考：用“通过 LDPC 的块”还原全局码字 → 抽取该 OFDM 的 Nd*2 位 → QPSK 映射成 Nd 维星座
        if len(promotable):
            # 未确定(-1)的信息位用 0 填充，然后整体扰码+整体LDPC编码，保证与打包顺序一致
            info_all = info_blocks.copy()
            info_all[info_all < 0] = 0
            info_flat = info_all.reshape(-1).astype(np.int8)
            bits_ldpc_hat, _ = ldpc_encode_bits(info_flat, c=code)  # 长度 = num_blocks * code.N

            for o in promotable:
                idxs = ofdm2flat[int(o)]
                if idxs.size < Nd * 2:
                    continue
                pair_bits = bits_ldpc_hat[idxs[:Nd * 2]].reshape(-1, 2)  # [Nd, 2]
                const_ref = QPSK_mapping(pair_bits)  # [Nd]
                pos2ref[int(o)] = const_ref

        # 生成与 pilot_pos 对齐的 pilot_ref_comb_fd（Nd 列，dtype=complex）
        if pilot_pos.size:
            pilot_ref_comb_fd = np.stack([pos2ref[int(p)] for p in pilot_pos], axis=0).astype(np.complex128)
        else:
            pilot_ref_comb_fd = np.zeros((0, Nd), dtype=np.complex128)

        # 进入下一轮
        iter_pseudo += 1
        continue
    end = time.time()
    print(f"耗时：{end-start:.4f}")

    # 与发端一致：收端解码后再加扰，得到最终位流（含 64bit 头）
    if args.use_scrambler:
        decoded_bits_raw = scramble_bits(info_blocks.flatten(),
                                         seed=args.scrambler_seed,
                                         mode=args.scrambler_mode,
                                         bit_width=args.scrambler_bitwidth).ravel()
    else:
        decoded_bits_raw = info_blocks.ravel()

    if groundtruth and plot and plot_opt['BER_show']:
        blk_bit_len = gt_bits_raw.size//code.K*code.K
        post_ber_per_blk = np.mean(
            decoded_bits_raw[:blk_bit_len].reshape(-1, code.K) != gt_bits_raw[:blk_bit_len].reshape(-1, code.K),
            axis=1
        )
        plot_pre_post_ber(post_ber=post_ber_per_blk)

    if head_bit:
        file_bits = 0
        type_bit = decoded_bits_raw[:args.type_bit_w]
        size_bit = decoded_bits_raw[args.type_bit_w: head_bit]
        for b in size_bit:
            file_bits = (file_bits << 1) | int(b)
        decoded_bits_raw = decoded_bits_raw[head_bit:head_bit+file_bits]
        if args.type_bit_w:
            type_bytes = np.packbits(type_bit)
            type_str = "".join(list(map(chr, type_bytes)))


    info = {
        "M": int(M),
        "ldpc_iter": iter_pseudo,
        "post_ber": np.mean(decoded_bits_raw != gt_bits_raw[head_bit:]) if groundtruth else None,
        'type_suffix': type_str if getattr(args, 'type_bit_w', 0) else None
    }

    return decoded_bits_raw, info
