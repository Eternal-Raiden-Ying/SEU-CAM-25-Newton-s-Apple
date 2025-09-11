from typing import Optional
import numpy as np
from .math_process import phase_unwrap_auto, fitting_line
from .DeltaInterpolator import DeltaInterpolator

__all__ = ['build_segments_from_pilots']


def _fit_drift_between(H_start, H_end, gap, N, * ,
                       symbol_len=None, return_phi=False,
                       plot: bool | int = False, DATA_BINS: np.ndarray | None = None):
    """
    用两个时间点（相隔 gap 个 OFDM）的信道估计做比值，拟合得到“每 OFDM”的
    频偏斜率 delta(SFO/CFO 残差) 以及常相位步进 phi（CPE）。
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

def _make_segments(pilot_pos: np.ndarray, M: int, start_idx: int | None):
    """
    把 comb pilot 位置拆成连续段：
      返回 list[(seg_L, seg_R, j_idx)]，其中 seg 是 (seg_L, seg_R]（左开右闭）索引区间的 data，
      j_idx 是右端的 comb 在 pilot_pos 中的索引（-1 表示尾段无右端 comb）。
    """
    segs = []
    prev = -1 if start_idx is None else int(start_idx)
    if pilot_pos.size == 0:
        segs.append((prev, M, -1))
        return segs
    for j, p in enumerate(pilot_pos):
        segs.append((prev, int(p), j))
        prev = int(p)
    if prev < M - 1:
        segs.append((prev, M, -1))  # 尾段
    return segs


def _fit_params_per_segment(H_start: np.ndarray, Hf_comb: np.ndarray, segs, *,
                            DATA_BINS: np.ndarray, symbol_len: int, N: int,
                            delta_global: float, phi_global: float):
    """
    对每个段 (prev, end] 拟合 (delta, phi)；并给出段端点的 H 引用：
      - H_pre: 段“起点参考”（上一 comb 或前导 H_start）
      - H_end: 段“右端 comb”的 H（无 comb 时为起点参考）
    """
    params = []
    H_pre = H_start.copy()
    last_d, last_p = float(delta_global), float(phi_global)
    for (prev_idx, end_idx, j_idx) in segs:
        gap = int(end_idx - prev_idx)
        if j_idx >= 0:
            H_end = Hf_comb[j_idx].copy()
            # 用起点参考与右端 comb 拟合每符号相位斜率 / 公共相位步进
            d_j, p_j = _fit_drift_between(H_pre, H_end, gap, N=N, symbol_len=symbol_len,
                                          return_phi=True, DATA_BINS=DATA_BINS,plot=False)
            last_d, last_p = float(d_j), float(p_j)
        else:
            H_end = H_pre.copy()
            d_j, p_j = last_d, last_p  # 尾段沿用上一段的估计或全局回退
        params.append({
            "prev_idx": int(prev_idx),
            "end_idx":  int(end_idx),
            "j_idx":    int(j_idx),
            "H_prev":   H_pre.copy(),
            "H_end":    H_end.copy(),
            "delta":    float(d_j),
            "phi":      float(p_j),
            "gap":      int(gap),
        })
        # 下一段起点参考更新为当前右端 comb
        if j_idx >= 0:
            H_pre = H_end.copy()
    return params


def _build_delta_interpolator(seg_params, *, symbol_len: int,
                              M: int, start_idx: int | None,
                              delta_global: float, interp_mode: str, interp_smooth: float) -> DeltaInterpolator:
    """
    用每段“大 delta”构造插值器的节点。
    这里选取“段右端索引”为节点位置（与 comb 对齐）；首段用 start_idx（或 -1）作为左端节点，尾段用 M-1。
    """
    interp = DeltaInterpolator(method=('hold' if interp_mode == 'hold' else
                                       'linear' if interp_mode == 'linear' else
                                       'cubic'))
    interp.set_params(smooth=interp_smooth)

    # 若无段，返回全 0
    if not seg_params:
        interp.add_node(0, 0.0, symbol_len)
        interp.build()
        return interp

    # 起点节点
    left_idx = -1 if start_idx is None else int(start_idx)
    first_delta = float(seg_params[0]["delta"]) if seg_params else 0.0
    interp.add_node(left_idx, first_delta, symbol_len)

    # 逐段在右端（comb处）挂节点
    for d in seg_params:
        if d["j_idx"] >= 0:
            interp.add_node(int(d["end_idx"]), float(d["delta"]), symbol_len)

    # 尾节点：若最后一段无右端 comb，用全局 delta 在 M-1 挂点
    last = seg_params[-1]
    if last["j_idx"] < 0:
        interp.add_node(int(M - 1), float(last.get("delta", delta_global)), symbol_len)

    interp.build()
    return interp


def _format_per_symbol_outputs(seg_params, *,
                               data_pos: np.ndarray, pilot_pos: np.ndarray,
                               DATA_BINS: np.ndarray, symbol_len: int, N: int,
                               distance_power: float, eps: float,
                               q_comb: np.ndarray | None):
    """
    把“段级参数 + 插值器”格式化为 **逐 data 符号**的输出（不会受 data_pos 不连续影响）：
      - 对每个 i：找到其左右 pilot（L, R），精确计算 L→i 与 i→R 的区间平均小段 delta。
      - 权重 w1/w2 按 距离×质量 计算。
      - 还原 H_start_per_sym / H_near_per_sym（用于外推融合）。
    """
    # 预先构造插值器
    interp = _build_delta_interpolator(seg_params, symbol_len=symbol_len,
                                       M=int(max(pilot_pos.max() if pilot_pos.size else 0, data_pos.max() if data_pos.size else 0) + 1),
                                       start_idx=seg_params[0]["prev_idx"] if seg_params else -1,
                                       delta_global=float(seg_params[-1]["delta"] if seg_params else 0.0),
                                       interp_mode='hold',  # 你可以把这里改为传入的 interp_mode
                                       interp_smooth=0.0)

    # 整理 comb 质量表：与 pilot_pos 对齐
    if q_comb is not None and pilot_pos.size:
        q = np.clip(q_comb.astype(float), 1e-3, 1.0)
    else:
        q = None

    # 输出容器
    T = data_pos.size
    Hs_out  = np.zeros((T, N), dtype=np.complex128)
    Hn_out  = np.zeros((T, N), dtype=np.complex128)
    d1_out  = np.zeros(T, dtype=float)   # dt from start
    d2_out  = np.zeros(T, dtype=float)   # |dt from near|
    dst_out = np.zeros(T, dtype=float)   # L->i 区间平均小段 delta
    den_out = np.zeros(T, dtype=float)   # i->R 区间平均小段 delta
    phi_out = np.zeros(T, dtype=float)   # 段的 phi（沿用右端 comb 对应段的 phi）
    w1_out  = np.zeros(T, dtype=float)
    w2_out  = np.zeros(T, dtype=float)

    # 为每个 data i 做“独立区间”计算，避免 mask 跨段累加错误
    for t, i in enumerate(data_pos):
        # 找左右 comb
        lefts  = pilot_pos[pilot_pos <= i]
        rights = pilot_pos[pilot_pos >= i]
        if lefts.size == 0 and rights.size == 0:
            # 极端情况：无任何 comb；退化为全局（取第一段的 H_ref/参数）
            d = seg_params[0] if seg_params else {"H_ref": np.ones(N, complex), "phi": 0.0, "delta": 0.0}
            Hs_out[t] = d["H_ref"]
            Hn_out[t] = d["H_ref"]
            d1_out[t] = 0.0; d2_out[t] = 0.0
            dst_out[t] = float(d["delta"]); den_out[t] = float(d["delta"])
            phi_out[t] = float(d["phi"])
            w1_out[t] = 1.0; w2_out[t] = 0.0
            continue

        L = int(lefts[-1]) if lefts.size else int(seg_params[0]["prev_idx"])
        R = int(rights[0]) if rights.size else int(seg_params[-1]["end_idx"])

        # 找到覆盖 i 的段（右端 comb 所在段）
        seg_idx = None
        for k, d in enumerate(seg_params):
            if d["prev_idx"] < i <= d["end_idx"]:
                seg_idx = k; break
        if seg_idx is None:
            seg_idx = len(seg_params) - 1
        dseg = seg_params[seg_idx]

        # 起点/近端参考
        Hs_out[t] = dseg["H_ref"]
        Hn_out[t] = dseg["H_near"]

        # 距离
        dt1 = float(max(i - dseg["prev_idx"], 0))
        dt2 = float(abs(i - dseg["end_idx"]))
        d1_out[t] = dt1
        d2_out[t] = dt2

        # 使用插值器精确计算 L->i 与 i->R 的“平均小段 delta”
        dst_out[t] = interp.mean_delta_over_interval(L, i, symbol_len=symbol_len) if i > L else float(dseg["delta"])
        den_out[t] = interp.mean_delta_over_interval(i, R, symbol_len=symbol_len) if R > i else float(dseg["delta"])

        phi_out[t] = float(dseg["phi"])

        # 质量（若可用则取 R 对应 comb 的质量）
        if q is not None and dseg["j_idx"] is not None and dseg["j_idx"] >= 0 and dseg["j_idx"] < q.size:
            qj = float(q[dseg["j_idx"]])
        else:
            qj = 1.0

        # 距离/质量权重
        w1 = 1.0 / np.power(dt1 + eps, distance_power)
        w2 = qj  * (1.0 / np.power(dt2 + eps, distance_power))
        s = w1 + w2
        w1_out[t] = w1 / s
        w2_out[t] = w2 / s

    return {
        "H_start_per_sym":  Hs_out,         # 每个 data 符号的“起点参考”H（段左端参考）
        "H_near_per_sym":   Hn_out,         # 每个 data 符号的“近端参考”H（段右端 comb）
        "delta_Li_per_sym": dst_out,        # L->i 区间平均小段 delta（修复不连续问题的关键）
        "delta_iR_per_sym": den_out,        # i->R 区间平均小段 delta
        "phi_per_sym":      phi_out,        # 段内 phi（右端 comb 对应段的 phi）
        "dt_from_start":    d1_out,         # 距离（左）
        "dt_from_near":     d2_out,         # 距离（右）
        "w1_per_sym":       w1_out,
        "w2_per_sym":       w2_out,
    }


# ======== 重构后的主函数 ========
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
                               # 若未提供 q_comb，则可在外部预先评估并传入；否则内部按 1 处理
                               symbols_comb_td: Optional[np.ndarray] = None,
                               pilot_ref_comb_fd: Optional[np.ndarray] = None,
                               # 权重控制
                               distance_power: float = 1.0,
                               eps: float = 1.0,
                               # 插值控制（method: 'hold' | 'linear' | 'cubic'）
                               interp_smooth: float = 0.0,
                               interp_mode: str = 'hold',
                               interp_plot: bool = False
                               ):
    """
    以“起点参考 + 若干 comb”分段，构建 data_pos 上逐符号的外推融合参数，并且
    通过“频偏插值 → 小段 delta 聚合”的方式，**正确支持 data_pos 不连续**。

    返回：
      pilot_seg: dict
        - "H_start": np.ndarray [N]                起点参考（前导最后一帧）
        - "H_comb":  np.ndarray [n_comb, N]        comb 参考
        - "pilot_pos": np.ndarray [n_comb]         comb 位置

      data_seg: dict
        - "H_start_per_sym": np.ndarray [T, N]     每个 data 符号起点参考 H
        - "H_near_per_sym":  np.ndarray [T, N]     每个 data 符号近端参考 H
        - "delta_Li_per_sym": np.ndarray [T]       L->i 平均小段 delta
        - "delta_iR_per_sym": np.ndarray [T]       i->R 平均小段 delta
        - "phi_per_sym":       np.ndarray [T]      段内 phi（与右端 comb 对齐）
        - "dt_from_start":     np.ndarray [T]
        - "dt_from_near":      np.ndarray [T]
        - "w1_per_sym":        np.ndarray [T]
        - "w2_per_sym":        np.ndarray [T]
    """
    # ------- 0) 规范化输入 -------
    pilot_pos = np.asarray(pilot_pos, dtype=int)
    all_idx   = np.arange(M, dtype=int)
    data_pos  = np.setdiff1d(all_idx, pilot_pos) if data_pos is None else np.asarray(data_pos, dtype=int)
    data_pos  = np.asarray(sorted(np.unique(data_pos))).astype(int)

    # ------- 1) 按 comb 分段 -------
    segs = _make_segments(pilot_pos, M=int(M), start_idx=start_idx)

    # ------- 2) 每段拟合 (delta, phi) 与端点 H -------
    seg_params = _fit_params_per_segment(H_start, Hf_comb, segs,
                                         DATA_BINS=DATA_BINS, symbol_len=symbol_len, N=N,
                                         delta_global=delta_global, phi_global=phi_global)

    # ------- 3) （可选）画插值节点（调试用） -------
    if interp_plot:
        import matplotlib.pyplot as plt
        xs = [d["end_idx"] for d in seg_params]
        ys = [d["delta"]   for d in seg_params]
        plt.figure(); plt.title("segment delta nodes"); plt.plot(xs, ys, 'o-'); plt.show()

    # ------- 4) 逐 data 符号生成外推融合所需量（修复不连续问题） -------
    per_sym = _format_per_symbol_outputs(seg_params,
                                         data_pos=data_pos, pilot_pos=pilot_pos,
                                         DATA_BINS=DATA_BINS, symbol_len=symbol_len, N=N,
                                         distance_power=distance_power, eps=eps,
                                         q_comb=q_comb)

    # ------- 5) 组装输出 -------
    pilot_seg = {
        "H_start":    H_start.copy(),
        "H_comb":     Hf_comb.copy(),
        "pilot_pos":  pilot_pos.copy(),
    }

    data_seg = per_sym  # 已含全部 per-symbol 字段
    return pilot_seg, data_seg