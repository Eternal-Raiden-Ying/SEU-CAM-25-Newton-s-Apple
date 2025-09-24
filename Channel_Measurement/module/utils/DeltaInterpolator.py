import warnings

import numpy as np
from .math_process import segment_means_on


__all__ = ['DeltaInterpolator']

class DeltaInterpolator:
    """
    面向对象的 delta 插值器：
      (1) update(idx_start, idx_end, delta)    —— 添加节点（会转换为频偏并缓存）
      (2) set_params(method, smooth)          —— 设置插值方式/参数
      (3) build()                              —— 构建插值器
      (4) delta_at(indices, symbol_len)        —— 返回这些整点处的“小段 delta”（由频偏反变换）
      (5) mean_delta_over_interval(a, b)       —— 区间 [a, b) 的平均小段 delta（整型边界）
    备注：
      - 小段 delta 的定义：每个符号跨度的一阶相位斜率，近似用“该整点处频偏 × symbol_len”。
      - 平均小段 delta：对 [a, b) 所跨的小段逐一取 delta，然后做平均。
      - 允许方法：'hold'（阶梯保持）、'linear'、'cubic'（CubicSpline）。默认 'hold'。
    """
    def __init__(self, fs=48000, method: str = 'hold', smooth: float = 0.0):
        self.fs: float = float(fs)
        self.method = method
        self.smooth = smooth
        self.node_pos: np.ndarray = np.array([])        # correspond to seg_fso
        self.seg_fso: np.ndarray = np.array([])         # interpolate res
        self._seg_n_s: list[int] = []
        self._seg_n_e: list[int] = []
        self._seg_dlt: list[float] = []
        self._seg_fso: np.ndarray = np.array([])        # known fso
        self._endpoint_idx: np.ndarray = np.array([])   # correspond to _seg_fso
        self._built = False

    def set_params(self, *, method: str | None = None, smooth: float | None = None, node_pos: np.ndarray | None):
        if method is not None:
            self.method = method
        if smooth is not None:
            self.smooth = float(smooth)
        if node_pos is not None:
            self.node_pos = node_pos
        self._built = False

    def update(self, idx_s: int | np.ndarray, idx_e: int | np.ndarray, delta: float | np.ndarray):
        self._built = False

        if isinstance(idx_s, np.ndarray):
            assert isinstance(idx_e, np.ndarray) and isinstance(delta, np.ndarray)
            assert idx_s.shape == idx_e.shape and delta.shape == idx_s.shape
            for index in range(idx_s.size):
                start = int(idx_s.ravel()[index])
                end = int(idx_e.ravel()[index])
                dlt = float(delta.ravel()[index])
                if start in self._seg_n_s:
                    i = self._seg_n_s.index(start)  # the first index, correspond to coarse-grained delta
                    if self._seg_n_e[i] <= end:
                        return
                    else:
                        self._seg_n_s.pop(i)
                        self._seg_n_e.pop(i)
                        self._seg_dlt.pop(i)
                self._seg_n_s.append(start)
                self._seg_n_e.append(end)
                self._seg_dlt.append(dlt)
        else:
            start = int(idx_s)
            end = int(idx_e)
            dlt = float(delta)
            if start in self._seg_n_s:
                i = self._seg_n_s.index(start)  # the first index, correspond to coarse-grained delta
                if self._seg_n_e[i] <= end:
                    return
                else:
                    self._seg_n_s.pop(i)
                    self._seg_n_e.pop(i)
                    self._seg_dlt.pop(i)
            self._seg_n_s.append(start)
            self._seg_n_e.append(end)
            self._seg_dlt.append(dlt)

        self._delta2fso()

    def _delta2fso(self):
        idx = np.argsort(np.array(self._seg_n_s))
        self._seg_fso = self.fs / (np.array(self._seg_dlt)[idx] + 1) - self.fs
        self._endpoint_idx = np.concatenate([
            np.array(self._seg_n_s)[idx], np.array(self._seg_n_e)[idx][-1][None]
        ], axis=0
        )
        return self._seg_fso

    def _build(self):
        if self._built: return
        self.seg_fso = segment_means_on(
            freq_offsets=np.concatenate([self._seg_fso, self._seg_fso[-1][None]]),
            ofdm_idx=np.concatenate([self._endpoint_idx, (self.node_pos[-1]+1)[None]]),
            eval_idx=self.node_pos,
            smoothing=self.smooth,
            extrap=self.method
        )
        self._built = True

    def _fso2delta(self):
        if not self._built:
            self._build()
        return 1 / (self.seg_fso / self.fs + 1) - 1

    def get_interp_delta(self, data_pos: int | np.ndarray, pilot_pos: int):
        delta_per_seg = self._fso2delta()
        pilot_idx = int(np.where(self.node_pos == pilot_pos)[0])
        if isinstance(data_pos, np.ndarray):
            deltas = []
            for element in data_pos:
                data_idx = int(np.where(self.node_pos == element)[0])
                if data_idx > pilot_idx:
                    deltas.append(np.sum(delta_per_seg[pilot_idx:data_idx]) / (data_idx - pilot_idx))
                elif data_idx < pilot_idx:
                    deltas.append(np.sum(delta_per_seg[data_idx:pilot_idx]) / (pilot_idx - data_idx))
                else:
                    raise ValueError(f"unexcepted idx, data pos {element} == pilot pos {pilot_pos}")
            return np.array(deltas)
        else:
            data_pos = int(data_pos)
            if data_pos > pilot_idx:
                delta = np.sum(delta_per_seg[pilot_idx + 1:data_pos + 1]) / (data_pos - pilot_idx)
            elif data_pos < pilot_idx:
                delta = np.sum(delta_per_seg[data_pos + 1:pilot_idx + 1]) / (pilot_idx - data_pos)
            else:
                raise ValueError(f"unexcepted idx, data pos {data_pos} == pilot pos {pilot_pos}")
            return delta

    def plot(self):
        if not self._built:
            self._build()
        from matplotlib import pyplot as plt
        raw_x = np.concatenate([
            self._endpoint_idx[:-1].reshape(-1,1), self._endpoint_idx[1:].reshape(-1,1)
        ], axis=1).ravel()

        raw_y = np.repeat(self._seg_fso, 2)

        plt.plot(raw_x, raw_y, color='blue',
                 label='comb pilot', linestyle='dotted', linewidth=3, alpha=0.7)
        plt.plot((self.node_pos[:-1]+self.node_pos[1:])/2, self.seg_fso,
                 color='red', label='interpolate', linewidth=3, alpha=0.7)
        plt.scatter((self.node_pos[:-1]+self.node_pos[1:])/2, self.seg_fso,
                    marker='*', s=7, color='black', label="_nolegend_")
        plt.xlabel("OFDM symbol index")
        plt.ylabel('fs offset/Hz')
        plt.legend(loc='upper right')
        plt.show()