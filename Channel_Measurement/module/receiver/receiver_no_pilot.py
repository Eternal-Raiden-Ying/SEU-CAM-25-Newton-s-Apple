# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import argparse
from matplotlib import pyplot as plt

from ..utils.print_aid import print_padded, print_dict_values
from ..utils.demodulate import get_symbols, get_constellation
from ..utils.encode import ldpc_encode_bits, ldpc_make_code, scramble_bits
from ..utils.modulate import generate_chirp, serial_to_parallel, QPSK_mapping
from ..utils.synchronize import synchronize
from ..utils.decode import ldpc_decode_blocks
from ..utils.io_interface import get_bits_from_file
from ..utils.plot import (plot_correlation, plot_received_signal, plot_original_constellations,
                          plot_corrected_constellations, plot_data_constellations, plot_unwrap_phase_fitting,
                          plot_impulse_response, plot_snr_over_time, plot_snr_over_subcarrier, plot_pre_post_ber)

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
)


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
    data_start      = args.data_start
    data_tail       = args.data_tail
    INTERVAL        = args.INTERVAL
    comb_seed_base  = args.COMB_PILOT_SEED_BASE

    groundtruth     = args.groundtruth
    tx_file_path    = getattr(args, "tx_file_path", None)
    head_bit        = args.head_bit

    # modulate
    clockwise       = args.clockwise
    scr_seed        = args.scrambler_seed
    scr_mode        = args.scrambler_mode
    scr_bitwidth    = args.scrambler_bitwidth

    # LDPC
    ldpc_standard   = args.ldpc_standard
    ldpc_rate       = args.ldpc_rate
    ldpc_z          = args.ldpc_z
    ldpc_ptype      = args.ldpc_ptype
    ldpc_device     = args.ldpc_device
    ldpc_llr_clip   = args.ldpc_llr_clip
    ldpc_max_iter   = args.ldpc_max_iter
    ldpc_verbose    = args.ldpc_verbose
    ldpc_log_every  = args.ldpc_log_every
    ldpc_check_every= args.ldpc_check_every
    ldpc_microbatch = args.ldpc_microbatch
    ldpc_batch      = args.ldpc_batch
    ldpc_print_iter = args.ldpc_print_iter

    # PLOT
    plot            = args.plot
    plot_opt        = args.plot_opt

    # PRINT
    print_flag      = getattr(args, 'print_flag', False)
    print_len       = getattr(args, 'print_len', 64)
    print_pad       = getattr(args, 'print_pad', '-')

    symbol_len = N + cp_len
    POS_BINS   = np.arange(1, N//2)
    DATA_BINS  = POS_BINS[data_start:-data_tail]
    Nd         = DATA_BINS.size

    code = ldpc_make_code(
        standard=ldpc_standard, rate=ldpc_rate, z=ldpc_z, ptype=ldpc_ptype,
        device=ldpc_device, llr_clip=ldpc_llr_clip, max_iter=ldpc_max_iter,
        verbose=ldpc_verbose, log_every=ldpc_log_every, check_every=ldpc_check_every,
        microbatch=ldpc_microbatch, print_iter=ldpc_print_iter
    )

    # ---------------- 0) 时域同步（scipy.signal.correlate） ----------------
    if print_flag: print_padded("begin to synchronize", print_len, print_pad)
    chirp_tpl = generate_chirp(fs, duration=chirp_len, f_l=chirp_l, f_h=chirp_h)
    ofdm_start, corr, chirp_start = synchronize(rx, chirp_tpl, mode="full")
    if plot and plot_opt['correlation']:
        plot_correlation(corr=corr,
                         axvline_dict={'chirp_start':chirp_start+chirp_tpl.size-1,
                                       'ofdm_start':ofdm_start+chirp_tpl.size-1})
    if print_flag: print_padded("synchronize done", print_len, print_pad)

    # ---------------- 1) 前导 pilot：H(f) 与漂移/基准估计 + 质量评估 ----------------
    if print_flag: print_padded("begin to analyze front pilot", print_len, print_pad)
    rx_pilot_td = rx[ofdm_start : ofdm_start + num_pilot * (N + cp_len)]
    sym_pilot_td = get_symbols(rx_pilot_td, cp_len=cp_len, N=N)         # [num_pilot, N] 时域
    Hf_pilot = evaluate_H_f(sym_pilot_td, pilots_fd=pilot)              # [num_pilot, N]
    res_arg = estimate_drift_and_origin(Hf_pilot, N=N, symbol_len=symbol_len, return_plot_args=plot_opt['unwrap'], mode='total')
    delta0, phi0, origin_H_f = res_arg[0], res_arg[1], res_arg[2]
    freq_bias = fs / (delta0 + 1) - fs

    pilot_metrics = analyze_pilots(
        symbols_td=sym_pilot_td, pilot_ref=pilot, DATA_BINS=DATA_BINS, mode="front", clockwise=clockwise,
        Hf=correct_H_f(origin_H_f=origin_H_f, N=N, index=np.arange(num_pilot),
                       symbol_len=symbol_len, delta=delta0, fixed_phase_shift_factor=phi0)
    )

    if print_flag:
        print(f"delta:{delta0}")
        print(f"fixed_phase_shift_factor:{phi0}")
        print(f"fs of receiver - fs of emitter = {freq_bias}")
        print(f"front pilot metrics: \nindex    snr     ber")
        print_dict_values(pilot_metrics, ["snr_db_med","ber"], [f"pilot {i}" for i in range(num_pilot)])

    if plot and plot_opt['impulse_response']:
        plot_impulse_response(h_t=np.fft.ifft(origin_H_f), fs=fs)
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
                                      symbol_len=symbol_len, delta=delta0, fixed_phase_shift_factor=phi0,
                                      DATA_BINS=DATA_BINS)
    if plot and plot_opt.get('snr_time_pilot', False):
        plot_snr_over_time(pilot_metrics["snr_db_med"], title="Front Pilot SNR over OFDM symbols")
    if print_flag: print_padded(f"front pilot analysis done", print_len, print_pad)

    # ---------------- 2) 数据段切片 ----------------
    rx_data_td = rx[ofdm_start + num_pilot * (N + cp_len):]
    symbols_all_td = get_symbols(rx_data_td, N=N, cp_len=cp_len)        # [M_guess, N]
    M_guess = symbols_all_td.shape[0]
    # M_guess = 299

    if head_bit:
        if print_flag: print_padded("begin to analyze file head", print_len, print_pad)

        # ---------------- 3) 先解出 64-bit 头所需的最小数据 ----------------
        #   64bit 头定义：LDPC 解码后的信息比特，再通过 scrambler 的前 64 bit。
        #   先构造 LDPC 码，决定需要多少 ofdm 数据符号（考虑 comb）
        T = estimate_M_from_filesize(filesize_bytes=head_bit // 8, K=code.K, Ncw=code.N, Nd=Nd, modulation_bits=2,
                                     interval=INTERVAL)
        T = min(T, M_guess)                     # 防越界
        T = max(T, INTERVAL+1)

        idx_T = np.arange(T)
        pilot_pos_T = idx_T[(idx_T % (INTERVAL + 1)) == INTERVAL]   # comb 位置
        data_pos_T = idx_T[(idx_T % (INTERVAL + 1)) != INTERVAL]   # 数据符号位置

        # comb 参考与 H(f)
        n_comb_T = pilot_pos_T.size
        if n_comb_T:
            pilot_ref_comb_fd_T = np.stack([generate_comb_pilot_symbol(N, comb_seed_base + i) for i in range(n_comb_T)],axis=0)
            symbols_comb_T_td = symbols_all_td[pilot_pos_T]
            Hf_comb_T = evaluate_H_f(symbols_comb_T_td, pilot_ref_comb_fd_T)

            # 段构建（质量加权的“前导最后一块 + 最近 comb”）
            pilot_pred_param_T, seg_T = build_segments_from_pilots(
                H_start=Hf_pilot[-1], Hf_comb=Hf_comb_T, pilot_pos=pilot_pos_T, M=T,
                DATA_BINS=DATA_BINS, q_comb=None, mode="quality_distance",
                symbol_len=symbol_len, N=N, delta_global=delta0, phi_global=phi0,
                symbols_comb_td=symbols_comb_T_td, pilot_ref_comb_fd=pilot_ref_comb_fd_T
            )

            comb_metrics_T = analyze_pilots(
                symbols_td=symbols_comb_T_td, pilot_ref=pilot_ref_comb_fd_T, DATA_BINS=DATA_BINS,
                Hf=correct_H_f(origin_H_f=pilot_pred_param_T['H_start'], delta=pilot_pred_param_T['delta'],
                               fixed_phase_shift_factor=pilot_pred_param_T['phi'], index=pilot_pred_param_T['gap'],
                               N=N, symbol_len=symbol_len),
                mode="comb", clockwise=clockwise,
            )

            # 外推 H_used（两路 + 权重融合）
            H_used_from_start_T = correct_H_f(
                origin_H_f=seg_T["H_start_per_seg"], index=seg_T["dt_from_start_per_sym"], N=N, symbol_len=symbol_len,
                delta=seg_T["delta_per_seg"], fixed_phase_shift_factor=seg_T["phi_per_seg"])
            H_used_from_near_T  = correct_H_f(
                origin_H_f=seg_T["H_near_per_sym"], index=seg_T["dt_from_near_per_sym"], N=N, symbol_len=symbol_len,
                delta=seg_T["delta_per_seg_per_sym"], fixed_phase_shift_factor=seg_T["phi_per_seg_per_sym"])

            w1_T, w2_T = seg_T["w1_per_sym"], seg_T["w2_per_sym"]
            H_used_T = (H_used_from_start_T * w1_T[:, None] + H_used_from_near_T * w2_T[:, None]) / (w1_T[:, None] + w2_T[:, None] + 1e-12)
        else:
            H_used_T = correct_H_f(origin_H_f=Hf_pilot[-1], N=N, symbol_len=symbol_len,
                                   delta=delta0, fixed_phase_shift_factor=phi0,
                                   index=data_pos_T).reshape(-1,N)


        # 取数据符号，均衡 -> PLL（门限由 pll_snr_median 提供）
        sym_data_T_td = symbols_all_td[data_pos_T].reshape(-1,N)
        const_zf_T = get_constellation(sym_data_T_td, H_used_T, DATA_BINS=DATA_BINS)
        pll_snr_med_T = pll_snr_median(const_zf_T)
        const_pll_T = apply_cpe_pll_sequence(
            const_zf_T, pll_snr_med_T,
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
        # 噪声/收缩
        sigmas_T = robust_sigma(const_pll_T, per_sc=args.sig_trk_per_sc)
        Habs2_T = np.abs(H_used_T[:, DATA_BINS])**2
        const_mmse_T = mmse_shrinkage(const_pll_T, Habs2_T, sigmas_T)

        # LLR + 按 per-SC SNR(dB) 缩放
        llr_raw_T, stat_T = llr_from_constellation(const_mmse_T, llr_clip=ldpc_llr_clip, clockwise=clockwise)
        scale_sc_T = llr_scale_by_snr(stat_T["snr_db_per_sc"], lo=2.0, hi=10.0, min_scale=0.4, max_scale=1.0)
        llr_scaled_T = (llr_raw_T.reshape(-1, Nd, 2) * scale_sc_T[:, :, None]).reshape(-1, Nd*2)

        # 打包 LLR（形状需一致）
        ofdm_idx_T = np.repeat(data_pos_T[:, None], Nd*2, axis=1)
        freq_axis_full = np.linspace(0.0, fs, N, endpoint=False)
        sub_carr_freq_T = np.repeat(np.repeat(freq_axis_full[DATA_BINS], 2)[None, :], data_pos_T.size, axis=0)
        llr_blocks_T = pack_llr_blocks(ofdm_idx=ofdm_idx_T, sub_carr_freq=sub_carr_freq_T, llr=llr_scaled_T)

        # 解出头若干 codeword
        decoded_head, it_head, _, _ = ldpc_decode_blocks(
            llr_blocks=llr_blocks_T,
            code=code,
            groundtruth_bits=None,
            head_bytes=0,
            batch=np.ceil(DATA_BINS.size/code.N).astype(int)
        )
        # 头 64 bit 定义在“解码后再扰码”的比特流上
        decoded_head_scr = scramble_bits(decoded_head, seed=scr_seed, mode=scr_mode, bit_width=scr_bitwidth)
        head64 = decoded_head_scr[:64]
        file_bits = 0
        for b in head64:
            file_bits = (file_bits << 1) | int(b)
        file_bytes = int(np.ceil(file_bits / 8.0))

        # 用 64bit 头推断总 OFDM 数（含 comb）
        M_total = estimate_M_from_filesize(filesize_bytes=file_bytes, K=code.K, Ncw=code.N, Nd=Nd, modulation_bits=2,
                                           interval=INTERVAL)
        if print_flag: print_padded("file head analysis done", print_len, print_pad)


    M_total = int(min(M_total, M_guess)) if head_bit else M_guess

    # ---------------- 4) 基于 M_total 的完整处理 ----------------
    # Groundtruth
    if groundtruth:
        assert tx_file_path is not None, "groundtruth=True 需要提供 --tx_file_path"
        print(f"use groundtruth, tx_file_path: {tx_file_path}")
        gt_bits_raw = get_bits_from_file(tx_file_path)
        if head_bit:
            gt_bits_raw = np.concatenate([head64, gt_bits_raw])
        gt_bits_scr = scramble_bits(gt_bits_raw, seed=scr_seed, mode=scr_mode, bit_width=scr_bitwidth)
        gt_bits_ldpc, _ = ldpc_encode_bits(gt_bits_scr, c=code)

    all_idx = np.arange(M_total)
    pilot_pos = all_idx[(all_idx % (INTERVAL + 1)) == INTERVAL]
    data_pos = all_idx[(all_idx % (INTERVAL + 1)) != INTERVAL]
    if plot and plot_opt['received_signal']:
        plot_received_signal(rx, ofdm_start, num_pilot, N, cp_len, M_total)

    # comb 参考与 H(f)
    if print_flag: print_padded("begin to analyze comb pilot", print_len, print_pad)
    n_comb = pilot_pos.size
    pilot_ref_comb_fd = (np.stack(
        [generate_comb_pilot_symbol(N, comb_seed_base + i) for i in range(n_comb)], axis=0
    ) if n_comb else np.zeros((0, N), complex))
    symbols_comb_td = (symbols_all_td[pilot_pos] if n_comb else np.zeros((0, N), complex))
    Hf_comb = evaluate_H_f(symbols_comb_td, pilot_ref_comb_fd) if n_comb else np.zeros((0, N), complex)

    # 段构建
    pilot_pred_param, seg = build_segments_from_pilots(
        H_start=Hf_pilot[-1], Hf_comb=Hf_comb,
        pilot_pos=pilot_pos, M=M_total, DATA_BINS=DATA_BINS, q_comb=None, mode="quality_distance",
        symbol_len=symbol_len, N=N, delta_global=delta0, phi_global=phi0,
        symbols_comb_td=symbols_comb_td, pilot_ref_comb_fd=pilot_ref_comb_fd
    )

    comb_metrics = (analyze_pilots(
        symbols_td=symbols_comb_td, pilot_ref=pilot_ref_comb_fd,
        DATA_BINS=DATA_BINS, mode="comb", clockwise=clockwise,
        Hf=correct_H_f(origin_H_f=pilot_pred_param['H_start'], delta=pilot_pred_param['delta'],
                       fixed_phase_shift_factor=pilot_pred_param['phi'], index=pilot_pred_param['gap'],
                       N=N, symbol_len=symbol_len)
        ) if n_comb else {"q": np.array([]), "snr_db_med": np.array([])})


    if plot and plot_opt['snr_time_comb']:
        plot_snr_over_time(comb_metrics["snr_db_med"], title="Comb Pilot SNR over OFDM symbols")
    if print_flag:
        print('index    delta           phi')
        print_dict_values(pilot_pred_param, ['delta', 'phi'],[f"{i}: pilot {index}" for i, index in enumerate(pilot_pos)])
        print('index      snr         ber')
        print_dict_values(comb_metrics, ['snr_db_med', 'ber'],[f"{i}: pilot {index}" for i, index in enumerate(pilot_pos)])
        print_padded("comb pilot analysis done", print_len, print_pad)

    # 外推 H_used（两路 + 权重融合）
    H_used_from_start = correct_H_f(
        origin_H_f=seg["H_start_per_seg"], index=seg["dt_from_start_per_sym"], N=N, symbol_len=symbol_len,
        delta=seg["delta_per_seg"], fixed_phase_shift_factor=seg["phi_per_seg"])
    H_used_from_near = correct_H_f(
        origin_H_f=seg["H_near_per_sym"], index=seg["dt_from_near_per_sym"], N=N, symbol_len=symbol_len,
        delta=seg["delta_per_seg_per_sym"], fixed_phase_shift_factor=seg["phi_per_seg_per_sym"])

    w1, w2 = seg["w1_per_sym"], seg["w2_per_sym"]
    H_used = (H_used_from_start * w1[:, None] + H_used_from_near * w2[:, None]) / (w1[:, None] + w2[:, None] + 1e-12)

    if print_flag: print_padded("begin to analyze data symbols", print_len, print_pad)
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

    # 噪声/收缩
    sigmas = robust_sigma(const_pll, per_sc=args.sig_trk_per_sc)
    Habs2 = np.abs(H_used[:, DATA_BINS])**2
    const_mmse = mmse_shrinkage(const_pll, Habs2, sigmas)
    if groundtruth:
        const_ref = QPSK_mapping(serial_to_parallel(gt_bits_ldpc, N=(Nd + 1) * 2))
        data_metrics = analyze_pilots(pilot_ref=const_ref, symbols_fd=const_mmse,DATA_BINS=np.arange(Nd),
                                      symbols_td=None, Hf=None,clockwise=clockwise, mode='data')
        print('index      snr         ber')
        print_dict_values(data_metrics, ['snr_db_med', 'ber'],
                          [f"{i}: data {index}" for i, index in enumerate(data_pos)])
        if plot and plot_opt['snr_time_data']:
            plot_snr_over_time(data_metrics["snr_db_med"], title="Data symbol SNR over OFDM symbols")
        if plot and plot_opt['data_constellation']:
            plot_data_constellations(const=const_mmse, data_pos=data_pos, const_ref=const_ref)
        if plot and plot_opt['snr_over_sc']:
            plot_snr_over_subcarrier(np.mean(snr_from_constellation(const_mmse, const_ref), axis=0),
                                     sc_idx=DATA_BINS, title="Average SNR over subcarriers (Data)")
    if print_flag: print_padded("data symbols analysis done", print_len, print_pad)

    # LLR + 按 per-SC SNR(dB) 缩放
    if print_flag: print_padded("begin to process and decode llr", print_len, print_pad)
    llr_raw, stat = llr_from_constellation(const_mmse, llr_clip=ldpc_llr_clip)
    scale_sc = llr_scale_by_snr(stat["snr_db_per_sc"], lo=2.0, hi=10.0, min_scale=0.4, max_scale=1.0)
    llr_scaled = (llr_raw.reshape(-1, Nd, 2) * scale_sc[:, :, None]).reshape(-1, Nd*2)

    # 打包 LLR（形状需一致）
    ofdm_idx = np.repeat(data_pos[:, None], Nd*2, axis=1)
    freq_axis_full = np.linspace(0.0, fs, N, endpoint=False)
    sub_carr_freq = np.repeat(np.repeat(freq_axis_full[DATA_BINS], 2)[None, :], data_pos.size, axis=0)
    llr_blocks = pack_llr_blocks(ofdm_idx=ofdm_idx, sub_carr_freq=sub_carr_freq, llr=llr_scaled)

    decoded_info, it, pre_ber, post_ber = ldpc_decode_blocks(
        llr_blocks=llr_blocks,
        code=code,
        groundtruth_bits=gt_bits_scr if groundtruth else None,
        head_bytes=head_bit//8,  # 64bit 头
        batch=ldpc_batch
    )

    if groundtruth and plot and plot_opt['BER_show']:
        plot_pre_post_ber(pre_ber, post_ber)

    # 与发端一致：收端解码后再加扰，得到最终位流（含 64bit 头）
    decoded_bits_scr = scramble_bits(decoded_info, seed=scr_seed, mode=scr_mode, bit_width=scr_bitwidth)
    if print_flag: print_padded("llr processing and decoding done", print_len, print_pad)

    info = {
        "M": int(M_total),
        "pilot_metrics": pilot_metrics,
        "comb_metrics": comb_metrics,
        "ldpc_iter": it,
        "pre_ber": np.mean(pre_ber) if args.groundtruth else None,
        "post_ber": np.mean(post_ber) if args.groundtruth else None,
        "data_pos": data_pos,
        "pilot_pos": pilot_pos,
    }
    if head_bit:
        decoded_bits_scr = decoded_bits_scr[head_bit: ]
    return decoded_bits_scr, info
