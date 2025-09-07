import warnings

import numpy as np
__all__ = ['phase_unwrap', 'fitting_line', 'normalize', 'phase_unwrap_auto', 'unique_sorted']

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


def smooth_H_moving_average(H_f, window_size=11):
    """
        few use
    :param H_f:
    :param window_size:
    :return:
    """
    window = np.ones(window_size) / window_size
    H_real = np.convolve(H_f.real, window, mode='same')
    H_imag = np.convolve(H_f.imag, window, mode='same')
    return H_real + 1j * H_imag


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
    neg_freq_mask = np.where(DATA_BINS >= N//2)[0]
    pos_freq_mask = np.where(DATA_BINS < N//2)[0]
    return np.concatenate([data[neg_freq_mask],data[pos_freq_mask]])


def phase_unwrap_auto(
    data: np.ndarray,
    *,
    # 幅度筛选：二选一（给 abs_thresh 就用绝对阈值；否则用分位数）
    abs_thresh: float | None = None,
    q_keep: float = 0.8,                 # 保留幅度最高的比例（quantile 模式）
    # unwrap 初值的 discont 候选集合
    discont_candidates = (np.pi, 1.05*np.pi, 1.1*np.pi, 1.2*np.pi, 1.3*np.pi, 1.4*np.pi, 1.5*np.pi, 1.7*np.pi),
    # discont_candidates = (np.pi,  1.1*np.pi, 1.3*np.pi, 1.4*np.pi, 1.5*np.pi, 1.7*np.pi),
    # 滑窗搜索配置（占比范围、最小样本数）
    win_frac_range = (0.05, 0.8),
    min_win_len: int = 50,
    # 是否对跨 0 频的窗口加一点偏好（通常更线性）
    center_bias: bool = True,
    # 展开后是否再做一次全段稳健微调
    refine_full_fit: bool = True,
    # 惩罚因子，用于防止斜率拟合结果过大，弃用中
    penal_factor: float = 1e3,
    penal_bound:  float = 1e-2,
    DATA_BINS: np.ndarray | None = None,
    N: int | None = None
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
        N = N = int(data.size)
        DATA_BINS = np.arange(N)

    if N == 0:
        warnings.warn("auto unwrap got no data")
        return np.array([], dtype=int), np.array([]), {'mode': 'empty'}

    # x: -N//2..N//2-1
    x_full = _centralize(data=np.arange(N), DATA_BINS=DATA_BINS, N=N)
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

if __name__ == "__main__":
    # here you can test these function if you are not familiar with them
    print()
