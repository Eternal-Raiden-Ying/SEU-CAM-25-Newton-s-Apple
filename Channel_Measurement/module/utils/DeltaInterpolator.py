import numpy as np
from scipy.interpolate import interp1d, CubicSpline

__all__ = ['DeltaInterpolator']

class DeltaInterpolator:
    """
    面向对象的 delta 插值器：
      (1) add_node(idx, delta, symbol_len)    —— 添加节点（会转换为频偏并缓存）
      (2) set_params(method, smooth)          —— 设置插值方式/参数
      (3) build()                              —— 构建插值器
      (4) delta_at(indices, symbol_len)        —— 返回这些整点处的“小段 delta”（由频偏反变换）
      (5) mean_delta_over_interval(a, b)       —— 区间 [a, b) 的平均小段 delta（整型边界）
    备注：
      - 小段 delta 的定义：每个符号跨度的一阶相位斜率，近似用“该整点处频偏 × symbol_len”。
      - 平均小段 delta：对 [a, b) 所跨的小段逐一取 delta，然后做平均。
      - 允许方法：'hold'（阶梯保持）、'linear'、'cubic'（CubicSpline）。默认 'hold'。
    """
    def __init__(self, method: str = 'hold', smooth: float = 0.0):
        self.nodes_idx: list[int] = []
        self.nodes_dlt: list[float] = []
        self.nodes_fpo: list[float] = []  # 频偏（delta / symbol_len）
        self.method = method
        self.smooth = smooth
        self._built = False
        self._interp = None  # callable: x -> freq_offset

    def add_node(self, idx: int, delta: float, symbol_len: int):
        self.nodes_idx.append(int(idx))
        self.nodes_dlt.append(float(delta))
        self.nodes_fpo.append(float(delta) / float(symbol_len))
        # 排序保持单调
        order = np.argsort(self.nodes_idx)
        self.nodes_idx = list(np.asarray(self.nodes_idx)[order])
        self.nodes_dlt = list(np.asarray(self.nodes_dlt)[order])
        self.nodes_fpo = list(np.asarray(self.nodes_fpo)[order])
        self._built = False

    def set_params(self, *, method: str | None = None, smooth: float | None = None):
        if method is not None:
            self.method = method
        if smooth is not None:
            self.smooth = float(smooth)
        self._built = False

    def _build_impl(self):
        xs = np.asarray(self.nodes_idx, dtype=float)
        ys = np.asarray(self.nodes_fpo, dtype=float)
        if xs.size == 0:
            # 没有节点：退化为常数0频偏
            self._interp = lambda x: np.zeros_like(np.asarray(x, dtype=float))
        elif xs.size == 1 or self.method == 'hold':
            # 台阶保持：就近向前保持
            x0, y0 = float(xs[0]), float(ys[0])
            x_last, y_last = float(xs[-1]), float(ys[-1])
            def _hold(xx):
                xx = np.asarray(xx, dtype=float)
                yy = np.zeros_like(xx)
                yy[xx <= x0] = y0
                yy[xx >= x_last] = y_last
                mask_mid = (xx > x0) & (xx < x_last)
                if mask_mid.any():
                    # 找到每个点的左侧最近节点
                    idx = np.searchsorted(xs, xx[mask_mid], side='right') - 1
                    yy[mask_mid] = ys[idx]
                return yy
            self._interp = _hold
        elif self.method in ('linear',):
            from scipy.interpolate import interp1d
            self._interp = interp1d(xs, ys, kind='linear', fill_value='extrapolate', assume_sorted=True)
        elif self.method in ('cubic','spline'):
            from scipy.interpolate import CubicSpline
            # smooth 在 CubicSpline 不直接用；如需光滑可外设平滑样条
            self._interp = CubicSpline(xs, ys, bc_type='natural')
        else:
            raise ValueError(f"unknown interp method: {self.method}")
        self._built = True

    def build(self):
        self._build_impl()

    def _ensure(self):
        if not self._built or self._interp is None:
            self._build_impl()

    def freq_offset_at(self, indices: np.ndarray | list[int] | int):
        self._ensure()
        return self._interp(indices)

    def delta_at(self, indices: np.ndarray | list[int] | int, symbol_len: int):
        fpo = self.freq_offset_at(indices)  # 频偏
        return np.asarray(fpo, dtype=float) * float(symbol_len)

    def mean_delta_over_interval(self, a: int, b: int, *, symbol_len: int) -> float:
        """
        区间 [a, b) 的平均“小段 delta”；若 a>=b 返回 0.
        """
        if b <= a:
            return 0.0
        # 用整点处小段delta近似 [a, b) 里每个单位间隔的小段
        samples = np.arange(a, b, dtype=int)
        dlt = self.delta_at(samples, symbol_len=symbol_len)  # len = b-a
        return float(np.mean(dlt)) if dlt.size else 0.0
