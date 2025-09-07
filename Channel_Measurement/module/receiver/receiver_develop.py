# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import argparse
from matplotlib import pyplot as plt

from ..utils.print_aid import print_padded, print_dict_values
from ..utils.math_process import unique_sorted
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


def contiguous_bounds(a):
    if len(a) == 0:
        return np.array([], dtype=a.dtype)

    # 找到不连续的位置
    breaks = np.where(np.diff(a) > 1)[0] + 1

    # 每一段的起点和终点
    starts = np.r_[a[0], a[breaks]]
    ends = np.r_[a[breaks - 1], a[-1]]

    # 拼接结果，避免重复
    result = []
    for s, e in zip(starts, ends):
        if s == e:
            result.append(s)
        else:
            result.extend([s, e])

    return np.array(result, dtype=a.dtype)


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
    M_guess = 215

    # TODO: develop时改动了部分函数的接口，仿照data symbol的处理修改提取head bit时的处理
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
        llr_blocks_T = pack_llr_blocks(ofdm_idx=ofdm_idx_T, sub_carr_freq=sub_carr_freq_T, llr=llr_scaled_T, code_N=code.N)

        # 解出头若干 codeword
        decoded_head, it_head, _, _, _ = ldpc_decode_blocks(
            llr_blocks=llr_blocks_T,
            code=code,
            groundtruth_bits=None,
            head_bytes=0,
            batch=ldpc_batch
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
    if plot and plot_opt['received_signal']:
        plot_received_signal(rx, ofdm_start, num_pilot, N, cp_len, M_total)

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
        const_data_global_ref = QPSK_mapping(serial_to_parallel(gt_bits_ldpc, N=(Nd + 1) * 2))

    all_idx = np.arange(M_total)
    pilot_pos_global = all_idx[(all_idx % (INTERVAL + 1)) == INTERVAL] if INTERVAL is not None else np.zeros(0)
    data_pos_global = all_idx[(all_idx % (INTERVAL + 1)) != INTERVAL] if INTERVAL is not None else all_idx


    max_pseudo_iter = getattr(args, "pseudo_pilot_max_iter", 3)
    alpha_pseudo_H = getattr(args, "pseudo_pilot_alpha", 0.5)  # H 融合平滑系数 [0,1]
    verbose_pseudo = getattr(args, "pseudo_pilot_verbose", True)

    iter_pseudo = 0
    circle_flag = True
    pilot_pos = pilot_pos_global
    data_pos = data_pos_global
    # initialize pilot_comb_fd if using comb-type pilot
    pilot_ref_comb_fd = (np.stack(
        [generate_comb_pilot_symbol(N, comb_seed_base + i) for i in range(pilot_pos_global.size)], axis=0
    ) if pilot_pos_global.size else np.zeros((0, N), complex))

    decoded_info = []
    while circle_flag and (iter_pseudo < max_pseudo_iter):
        n_comb = pilot_pos.size
        symbols_comb_td = (symbols_all_td[pilot_pos] if n_comb else np.zeros((0, N), complex))
        Hf_comb = evaluate_H_f(symbols_comb_td, pilot_ref_comb_fd, DATA_BINS) if n_comb else np.zeros((0, N), complex)

        # 段构建
        pilot_pred_param, seg = build_segments_from_pilots(
            H_start=Hf_pilot[-1], Hf_comb=Hf_comb, pilot_pos=pilot_pos, data_pos=data_pos,
            M=M_total, DATA_BINS=DATA_BINS, q_comb=None, mode="quality_distance",
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
            plot_snr_over_time(comb_metrics["snr_db_med"], title="Comb Pilot SNR over OFDM symbols", pos=pilot_pos)
        if print_flag:
            print('index    delta           phi')
            print_dict_values(pilot_pred_param, ['delta', 'phi'],[f"{i}: pilot {index}" for i, index in enumerate(pilot_pos)])
            print('index      snr         ber')
            print_dict_values(comb_metrics, ['snr_db_med', 'ber'],[f"{i}: pilot {index}" for i, index in enumerate(pilot_pos)])


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

        # 噪声/收缩
        sigmas = robust_sigma(const_pll, per_sc=args.sig_trk_per_sc)
        Habs2 = np.abs(H_used[:, DATA_BINS])**2
        const_mmse = mmse_shrinkage(const_pll, Habs2, sigmas)
        if groundtruth:
            const_ref = const_data_global_ref[np.where(np.isin(data_pos_global, data_pos))[0]]
            data_metrics = analyze_pilots(pilot_ref=const_ref, symbols_fd=const_mmse,DATA_BINS=np.arange(Nd),
                                          symbols_td=None, Hf=None,clockwise=clockwise, mode='data')
            if print_flag:
                print('index      snr         ber')
                print_dict_values(data_metrics, ['snr_db_med', 'ber'], [f"{i}: data {index}" for i, index in enumerate(data_pos)])
            if plot and plot_opt['snr_time_data']:
                plot_snr_over_time(data_metrics["snr_db_med"], title="Data symbol SNR over OFDM symbols", pos=data_pos)
            if plot and plot_opt['data_constellation']:
                plot_data_constellations(const=const_mmse, data_pos=data_pos, const_ref=const_ref)
            if plot and plot_opt['snr_over_sc']:
                plot_snr_over_subcarrier(np.mean(snr_from_constellation(const_mmse, const_ref), axis=0),
                                         sc_idx=DATA_BINS, title="Average SNR over subcarriers (Data)")


        # LLR + 按 per-SC SNR(dB) 缩放
        llr_raw, stat = llr_from_constellation(const_mmse, llr_clip=ldpc_llr_clip)
        scale_sc = llr_scale_by_snr(stat["snr_db_per_sc"], lo=2.0, hi=10.0, min_scale=0.4, max_scale=1.0)
        llr_scaled = (llr_raw.reshape(-1, Nd, 2) * scale_sc[:, :, None]).reshape(-1, Nd*2)

        # TODO: 现有的逻辑仅适配原来无伪导频的处理模式，当某一llr_block跨过两个OFDM符号，
        #  且只有其中一个OFDM symbol是data symbol(则此llr_block实际在上一循环已经通过LDPC校验)，
        #  此时不再需要将对应的llr_block传给解码器，因此llr_scaled需要根据已经解码的情况作截断处理，
        #  另外ofdm_idx, sc_freq需要对应修改，匹配每一bit的llr，
        #  建议利用全局的ofdm_idx_global，sub_carr_freq_global(都需要自己新建)进行切片得到此处的参数，
        #  全局的decoded_info设计对应的visited_flag或者预先用nan填充以便于对llr_scaled进行切片
        ofdm_idx = np.repeat(data_pos[:, None], Nd*2, axis=1)
        freq_axis_full = np.linspace(0.0, fs, N, endpoint=False)
        sub_carr_freq = np.tile(np.repeat(freq_axis_full[DATA_BINS], 2), data_pos.size)
        llr_blocks = pack_llr_blocks(ofdm_idx=ofdm_idx, sub_carr_freq=sub_carr_freq, llr=llr_scaled, Ncw=code.N)

        # TODO: 将BER计算移出ldpc_decode_blocks的逻辑，将BER计算移到循环外，即当所有llr_block解码完毕再计算最后的BER并可视化
        decoded_info_part, it, pre_ber, post_ber, syn = ldpc_decode_blocks(
            llr_blocks=llr_blocks,
            code=code,
            groundtruth_bits=None,
            batch=ldpc_batch
        )
        if print_flag and verbose_pseudo:
            print_padded(f"[pseudo] iter {iter_pseudo}: syn_nonzero={np.count_nonzero(syn)}", print_len, print_pad)

        circle_flag = np.any(syn != 0)
        ok_llr = np.repeat(syn == 0, code.N)
        pb_ofdm_idx = unique_sorted(llr_blocks['ofdm_idx'].flatten()[ok_llr==0])  # ofdm symbol (problem) idx
        decoded_info_part = decoded_info_part.reshape(-1, code.K)
        # TODO: 对于通过LDPC校验的部分，逐llr_block写入解码后的信息，未通过LDPC校验的部分，位置保留，便于后续写入
        # update data_pos, pilot_pos, pilot_ref_comb_fd
        if pb_ofdm_idx.size == data_pos.size:
            # fail to decode new things
            # TODO: 当采取新的伪导频没有解出新的可靠信息时，将未通过LDPC校验的部分也写入decoded_info,结束循环
            circle_flag = False

        else:
            # Use the part that failed the LDPC check as the data symbol for the next iteration
            data_pos = np.array(pb_ofdm_idx)
            # Use the part that has already passed the LDPC check and the original comb pilot as the pilot symbol
            # for the next cycle. Only take the non-continuous parts to reduce unnecessary calculations.
            pilot_pos = contiguous_bounds(np.setdiff1d(all_idx, data_pos))

            # TODO: 生成新的pilot_ref_comb_fd,
            #       判断所需的pilot_pos中是否包含pilot_pos_global,如果包含，需要按照pilot index合并两部分
            #       另外，根据ldpc解码生成的pilot_ref_fd仅包含DATA_BINS部分，对应pilot_pos_global部分的pilot_ref需要用DATA_BINS截断，以构成二维数组

            # 进入下一轮
            iter_pseudo += 1
            continue  # while circle_flag 的下一轮：会据新的 pilot_pos/pilot_ref_comb_fd 重建 segment


    # TODO: groundtruth模式中，BER处理逻辑新增在这里

    if groundtruth and plot and plot_opt['BER_show']:
        plot_pre_post_ber(pre_ber, post_ber)
        plt.scatter(np.arange(syn.size),syn, s=1)
        plt.show()

    # 与发端一致：收端解码后再加扰，得到最终位流（含 64bit 头）
    decoded_bits_scr = scramble_bits(decoded_info, seed=scr_seed, mode=scr_mode, bit_width=scr_bitwidth)

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
