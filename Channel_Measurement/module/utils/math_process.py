import warnings
import numpy as np
from typing import Optional, Tuple
from scipy.interpolate import CubicSpline, UnivariateSpline

__all__ = ['phase_unwrap', 'fitting_line', 'normalize', 'phase_unwrap_auto',
           'unique_sorted', 'segment_means_on']

def unique_sorted(x):
    if len(x) == 0:
        return x
    # 保留第一个元素，以及相邻不相等的元素
    mask = np.concatenate(([True], np.diff(x) != 0))
    return x[mask]


def phase_unwrap(data:np.ndarray, estimate_start=0, estimate_percent=0.1,*,
                 initial_discont=1.5*np.pi, mode='truncated', amp_filter_th=1.2):
    """
        improved unwrap phase function
        first use default unwrap() to get a linear part and unwrap precisely based on that,
        so make sure arguments are set right to get the initial linear part,
        related arg: estimate_start, estimate_percent, initial_discont
    :param data: phase unwrapped (not only the angle, amplitude will be used to filter)
    :param estimate_start: start from where you get the linear part
    :param estimate_percent: length percent of data to estimate the linear part
    :param initial_discont: discont for default unwrap(), the larger discont is, the more func will tolerant noise
    :param x_filtered: if data is filtered, raw x val is needed to do polyfit
    :param amp_filter_th: Amplitude filter threshold, the more it nearer to 1, the more strict the filter is
    :param mode: unwrap mode, supported ['simple', 'filtered', 'truncated']
    :return: index, unwrapped phase
    """

    if mode == 'simple':
        x = np.arange(data.size)
        y = np.angle(data)
    elif mode == 'filtered':
        mask = np.where(np.abs(data) < amp_filter_th)
        x = np.arange(data.size)[mask]
        y = np.angle(data)[mask]
    elif mode == 'truncated':
        N = data.size
        data = data[:N//2]
        mask = np.where(np.abs(data) < amp_filter_th)
        x = np.arange(data.size)[mask]
        y = np.angle(data)[mask]
    else:
        raise ValueError("Unexpected mode, supported ['simple', 'filtered', 'truncated']")

    initial_res = np.unwrap(y, discont=initial_discont)
    estimate_len = int(y.size * estimate_percent)
    estimate_part = initial_res[estimate_start:estimate_start+estimate_len]
    coeffs = np.polyfit(x[estimate_start:estimate_start+estimate_len], estimate_part, deg=1)
    slope, intercept = coeffs
    fit_line = slope * x + intercept
    unwrap_refined = y + 2 * np.pi * np.round((fit_line - y) / (2 * np.pi))
    return x, unwrap_refined

def fitting_line(x, y,*, filter=True, residual_th=2):
    """
        fit a line for given x and y (y=kx+b), then return k,b
        package the residual filter, thus fitting could be more precise
    :param x:
    :param y:
    :param filter: boolean
    :param residual_th: use for mask, threshold = std * residual_th
    :return:
    """
    coeffs = np.polyfit(x,y,deg=1)
    if filter:
        fit_line = np.polyval(coeffs, x)
        residual = y - fit_line
        std = np.std(residual)
        mask = np.where(np.abs(residual) < residual_th * std)
        coeffs_refined = np.polyfit(x[mask], y[mask], deg=1)
        slope, intercept = coeffs_refined
    else:
        slope, intercept = coeffs

    return slope, intercept


def normalize(data: np.ndarray,*, axis=-1, keepdim=False):
    """
        normalize given data
    :param data:
    :return:
    """
    if data.ndim == 1:
        max_val = np.max(np.abs(data))
    else:
        max_val = np.max(np.abs(data), axis=axis, keepdims=keepdim)
    return data / max_val

def _angle(y):
    return np.angle(y) if np.iscomplexobj(y) else np.asarray(y, dtype=float)

def _circ_diff(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi

def _mad(x, c=1.4826):
    x = np.asarray(x)
    if x.size == 0:
        return np.inf
    med = np.median(x)
    return c * np.median(np.abs(x - med))

def _robust_line_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0, float(np.median(y)) if y.size else 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    # OLS
    s, b = np.linalg.lstsq(A, y, rcond=None)[0]
    # 一轮 Huber
    r = y - (s * x + b)
    s_hat = 1.345 * _mad(r) + 1e-12
    w = 1.0 / np.maximum(1.0, np.abs(r) / s_hat)
    Aw = A * w[:, None]
    yw = y * w
    s, b = np.linalg.lstsq(Aw, yw, rcond=None)[0]
    return s, b

def _centralize(data:np.ndarray, DATA_BINS:np.ndarray, N:int):
    assert data.ndim == 1
    if data.size > DATA_BINS.size:
        assert data.size == N
        data=data[DATA_BINS]
    elif data.size == DATA_BINS.size:
        pass
    else:
        raise ValueError(f"unexcepted shape, data.shape {data.shape}, DATA_BINS.shape {DATA_BINS.shape}")
    neg_freq_mask = np.where(DATA_BINS >= N//2)[0]
    pos_freq_mask = np.where(DATA_BINS < N//2)[0]
    return np.concatenate([data[neg_freq_mask],data[pos_freq_mask]])


def phase_unwrap_auto(
    data: np.ndarray,
    *,
    # 幅度筛选：二选一（给 abs_thresh 就用绝对阈值；否则用分位数）
    abs_thresh: float | None = None,
    q_keep: float = 0.7,                 # 保留幅度最高的比例（quantile 模式）
    # unwrap 初值的 discont 候选集合
    discont_candidates = (np.pi, 1.1*np.pi, 1.2*np.pi, 1.3*np.pi, 1.4*np.pi, 1.5*np.pi, 1.7*np.pi),
    # discont_candidates = (np.pi,  1.1*np.pi, 1.3*np.pi, 1.4*np.pi, 1.5*np.pi, 1.7*np.pi),
    # 滑窗搜索配置（占比范围、最小样本数）
    win_frac_range = (0.02, 0.8),
    min_win_len: int = 50,
    # 是否对跨 0 频的窗口加一点偏好（通常更线性）
    center_bias: bool = False,
    # 展开后是否再做一次全段稳健微调
    refine_full_fit: bool = True,
    # 惩罚因子，用于防止斜率拟合结果过大，弃用中
    penal_factor: float = 1e3,
    penal_bound:  float = 1e-2,
    DATA_BINS: np.ndarray | None = None,
    N: int | None = None,
    score_th: float = 0.5,
    score_gate_ctrl_leakage: bool = True
):
    """
    自适应一维相位展开（智能选线性片段 + 多discont择优 + 稳健拟合）
    default truncated

    返回：
        x: np.ndarray（索引）
        unwrap: np.ndarray（展开后的相位，与 y 同长度）
        meta: dict（过程信息，便于调试）
    """
    data = np.asarray(data)
    if DATA_BINS is not None:
        assert isinstance(N, int), "if using DATA_BINS, N must be explicitly given"
    else:
        N = int(data.size)
        DATA_BINS = np.arange(N)

    if N == 0:
        warnings.warn("auto unwrap got no data")
        return np.array([], dtype=int), np.array([]), {'mode': 'empty'}

    # x: -N//2..N//2-1

    x_full = _centralize(data=np.concatenate([np.arange(N//2),np.arange(N//2)-N//2]), DATA_BINS=DATA_BINS, N=N)
    # y: 先取角，再“把负频段挪前”
    y_full = _angle(data)
    y_full = _centralize(data=y_full, DATA_BINS=DATA_BINS, N=N)

    # ---------- 自适应幅度掩码 ----------
    amp_full = np.abs(y_full) if np.iscomplexobj(y_full) else np.ones_like(y_full)
    if abs_thresh is not None:
        mask = amp_full >= abs_thresh
    else:
        thr = np.quantile(amp_full, 1.0 - q_keep)
        mask = amp_full >= thr

    # 防止掩码过严
    if mask.sum() < max(min_win_len, 10):
        warnings.warn("unwrap did not get enough data from amp filter")
        mask = np.ones_like(mask, dtype=bool)

    x = x_full[mask]
    y = y_full[mask]

    # 极端情况：样本不足，退回一次简单 unwrap
    if y.size < 8:
        unwrap_simple = np.unwrap(y_full, discont=np.pi)
        warnings.warn("unwrap fallback!")
        return x_full, unwrap_simple, {'mode': 'fallback_small'}

    # ---------- 多 discont + 多尺度滑窗搜索 ----------
    Lmin = max(int(N * win_frac_range[0]), min_win_len)
    Lmax = max(Lmin, int(N * win_frac_range[1]))
    win_lengths = np.unique(np.linspace(Lmin, Lmax, num=20, dtype=int))

    best = {'score': np.inf}

    def _window_ok(xs):
        if not center_bias:
            return True
        # “跨 0 频”或包含 0 的窗口优先（更接近整体线性）
        return (xs.min() <= 0) and (xs.max() >= 0)

    stop_flag = False
    for d in discont_candidates:
        y_unw0 = np.unwrap(y, discont=d)

        for L in win_lengths:
            if x.size < L:
                continue
            step = max(1, L // 8)
            for start in range(0, x.size - L + 1, step):
                sl = slice(start, start + L)
                xs = x[sl]
                if not _window_ok(xs):
                    continue

                ys_unw = y_unw0[sl]
                # 在粗 unwrap 上做稳健线性拟合
                s, b = _robust_line_fit(xs, ys_unw)
                k = round(b / (2 * np.pi))
                b = b - - 2 * np.pi * k

                # 用原始相位(未 unwrap)的“圆残差 MAD”作为评分
                fit = s * x + b
                y_score = y + 2 * np.pi * np.round((fit - y) / (2 * np.pi))
                # score = (_mad(y_score - fit) + penal_factor * np.abs(s)) if np.abs(s) > penal_bound else _mad(y_score - fit)
                score = (_mad(y_score - fit))
                if score < best['score']:
                    best = {
                        'score': score,
                        'discont': d,
                        'slope': s,
                        'intercept': b,
                        'win_idx': x[start],
                        'win_len': L
                    }
                    if score_gate_ctrl_leakage:
                        if score < score_th:
                            stop_flag = True
                            break

        if stop_flag: break


    # 若依然不稳定，回退
    if not np.isfinite(best['score']):
        unwrap_simple = np.unwrap(y_full, discont=np.pi)
        return x_full, unwrap_simple, {'mode': 'fallback_unstable'}

    # ---------- 用最佳直线模型对齐全段 ----------
    k = round(best['intercept'] / (2 * np.pi))
    best['intercept'] = best['intercept'] - 2 * np.pi * k
    fit_line_full = best['slope'] * x_full + best['intercept']
    unwrap_full = y_full + 2 * np.pi * np.round((fit_line_full - y_full) / (2 * np.pi))

    # ---------- 可选：全段稳健微调 ----------
    if refine_full_fit:
        s2, b2 = _robust_line_fit(x_full, unwrap_full)
        fit2 = s2 * x_full + b2
        unwrap_full = y_full + 2 * np.pi * np.round((fit2 - y_full) / (2 * np.pi))
        best['slope_refined'] = float(s2)
        best['intercept_refined'] = float(b2)

    meta = {
        'chosen_discont': float(best['discont']),
        'win_start': int(best['win_idx']),
        'win_len': int(best['win_len']),
        'score_mad': float(best['score']),
        'slope': float(best['slope']),
        'intercept': float(best['intercept']),
        'center_bias': bool(center_bias)
    }
    if refine_full_fit:
        meta.update({
            'slope_refined': best['slope_refined'],
            'intercept_refined': best['intercept_refined']
        })

    return x_full, unwrap_full, meta


def _build_cumulative_spline(
    freq_offsets: np.ndarray,
    ofdm_idx: np.ndarray,
    smoothing: float = 0.0,
    preserve_endpoints: bool = True,
    endpoint_weight: float = 1e6,
    bc_type: str = "natural",
):
    """
    在累计量 F(x) 上拟合三次样条（x 以符号索引替代时间）。
    freq_offsets[k] 是 [ofdm_idx[k], ofdm_idx[k+1]) 的段均值。
    返回: 可在区间内外求值的 F(x) 样条对象。
    """
    freq_offsets = np.asarray(freq_offsets, dtype=float)
    ofdm_idx = np.asarray(ofdm_idx, dtype=float)

    L = freq_offsets.size
    if ofdm_idx.size != L + 1:
        raise ValueError("len(ofdm_idx) must be len(freq_offsets)+1")
    if not np.all(np.diff(ofdm_idx) > 0):
        raise ValueError("ofdm_idx must be strictly increasing")

    seg_len = np.diff(ofdm_idx)                # 段长度（符号数或等效时间）
    if np.any(seg_len <= 0):
        raise ValueError("Non-positive segment length found.")

    # 每段积分增量 ΔF_k = 段均值 * 段长
    delta_F = freq_offsets * seg_len

    # 边界累计量 F(t_i)
    F = np.zeros(L + 1, dtype=float)
    F[1:] = np.cumsum(delta_F)

    x = ofdm_idx

    if smoothing is None or smoothing <= 0:
        # 精确插值：严格保持原大段积分
        if bc_type.lower() == "clamped":
            # dF/dx=δfs；端点导数用首/末段均值近似
            bc = ((1, freq_offsets[0]), (1, freq_offsets[-1]))
        elif bc_type.lower() == "natural":
            bc = "natural"
        else:
            raise ValueError("bc_type must be 'natural' or 'clamped'")
        spline_F = CubicSpline(x, F, bc_type=bc, extrapolate=True)
    else:
        # 平滑样条：牺牲严格等式，换更平滑的 δfs
        w = np.ones_like(F)
        if preserve_endpoints:
            w[0] *= endpoint_weight
            w[-1] *= endpoint_weight
        spline_F = UnivariateSpline(x, F, w=w, s=float(smoothing), k=3)
        # UnivariateSpline 默认允许区间外求值

    return spline_F


def _interval_mean_with_extrap(
    spline_F, x_min: float, x_max: float, a: float, b: float, extrap: str
) -> float:
    """
    计算 [a,b] 上的平均值：(F(b)-F(a))/(b-a)，带外推策略。
    extrap='spline'  : 直接用样条在区间外求值
    extrap='hold'    : 区间外保持 δfs 常值（F 线性延拓）
    """
    if b <= a:
        raise ValueError("Empty or negative interval.")

    if extrap == "spline":
        return (spline_F(b) - spline_F(a)) / (b - a)

    elif extrap == "hold":
        # 预备：边界处的一阶导数（δfs）
        dF = spline_F.derivative(1)
        delta_left = dF(x_min)
        delta_right = dF(x_max)

        total = 0.0
        # 左侧区间外
        if a < x_min:
            left_end = min(b, x_min)
            total += delta_left * (left_end - a)
            a = left_end  # 把左外部分耗尽

        # 右侧区间外
        right_start = max(a, x_max)
        if b > x_max:
            # 先加中间部分（如果还有）
            if right_start > a:
                total += spline_F(right_start) - spline_F(a)
            # 再加右外部分
            total += delta_right * (b - max(a, x_max))
        else:
            # 完全在区间内（或左外已处理）
            if b > a:
                total += spline_F(b) - spline_F(a)

        return total / (b - (a if a < b else a))  # 分母是原始区间长度

    else:
        raise ValueError("extrap must be 'spline' or 'hold'")


def segment_means_on(
    freq_offsets: np.ndarray,
    ofdm_idx: np.ndarray,
    eval_idx: np.ndarray,
    smoothing: float = 0.0,
    preserve_endpoints: bool = True,
    endpoint_weight: float = 1e6,
    bc_type: str = "natural",
    extrap: str = "spline",  # 'spline' 或 'hold'
) -> np.ndarray:
    """
    在新的边界 eval_idx 上计算“新段均值”。允许 eval_idx 超出原始 ofdm_idx（外推）。

    参数
    ----
    freq_offsets : (L,)      原始大段的平均值
    ofdm_idx     : (L+1,)    原始大段边界（升序）
    eval_idx     : (M+1,)    新边界（升序；可超出原范围以触发外推）
    smoothing    : float     平滑参数；0 保积分严格，>0 更平滑（原段积分近似）
    bc_type      : str       'natural' 或 'clamped'（仅 s<=0 时）
    extrap       : str       'spline' 直接样条外推；'hold' 区间外保持 δfs 常值

    返回
    ----
    new_means : (M,)         每个 [eval_idx[k], eval_idx[k+1]] 的平均值
    """
    spline_F = _build_cumulative_spline(
        freq_offsets, ofdm_idx,
        smoothing=smoothing,
        preserve_endpoints=preserve_endpoints,
        endpoint_weight=endpoint_weight,
        bc_type=bc_type,
    )

    eval_idx = np.asarray(eval_idx, dtype=float)
    if eval_idx.ndim != 1 or eval_idx.size < 2:
        raise ValueError("eval_idx must have length >= 2.")
    if not np.all(np.diff(eval_idx) > 0):
        raise ValueError("eval_idx must be strictly increasing.+")

    x_min, x_max = float(ofdm_idx[0]), float(ofdm_idx[-1])
    a = eval_idx[:-1]
    b = eval_idx[1:]

    new_means = np.empty_like(a, dtype=float)
    for i in range(a.size):
        new_means[i] = _interval_mean_with_extrap(spline_F, x_min, x_max, a[i], b[i], extrap)

    return new_means


if __name__ == "__main__":
    # here you can test these function if you are not familiar with them
    print()
