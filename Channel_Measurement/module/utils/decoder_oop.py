# utils/decoder_oop.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple, List
import numpy as np

# 依赖你现有的工具函数（名称按你仓库中 utils/batch.py 来）
from .batch import (
    evaluate_H_f,            # Ĥ(f) 估计：evaluate_H_f(symbols_td, pilots_fd)
    correct_H_f,             # H 外推：correct_H_f(origin_H_f, delta, N, index, symbol_len, fixed_phase_shift_factor)
    _fit_drift_between,       # 拟合跨 gap 漂移（delta, phi_step）
    get_constellation,       # 均衡并抽取 DATA_BINS
    pll_snr_median,          # 逐符号中位 SNR(dB)
    _qpsk_hard,              # QPSK 硬判（用于噪声估计）
    mmse_shrinkage,
    robust_sigma
)

# ========== 配置结构 ==========
@dataclass
class SigmaTrackerConfig:
    per_sc: bool = True       # True=每子载波跟踪 σ；False=整符号一个 σ
    alpha_min: float = 0.05   # EMA 最小学习率
    alpha_max: float = 0.25   # EMA 最大学习率
    init_sigma: float = 0.15  # 初始 σ（按你数据尺度适当调整）
    snr_mid_db: float = 6.0   # 把 SNR 压到 [0,1] 的中点
    snr_scale: float = 4.0    # 压缩斜率


@dataclass
class PLLConfig:
    alpha: float = 0.15
    snr_th_db: float = 6.0
    alpha_min: float = 0.05
    alpha_max: float = 0.30
    snr_th_min_db: float = 3.0
    snr_th_max_db: float = 10.0
    beta: float = 0.9
    snr_mid_db: float = 6.0
    snr_scale: float = 4.0


@dataclass
class DriftGuardConfig:
    enabled: bool = False        # 开关
    n_sigma: float = 3.0        # “均值±nσ”的 n
    fit_window: int = 5         # 外推/统计窗口（最近 K 个被接受的样本）
    hist_max: int = 200         # 历史缓存上限，防止无限增长
    warn_on_replace: bool = True  # ← 发生替换时打印告警
    warn_prefix: str = "[DriftGuard]"  # ← 告警前缀


@dataclass
class DecoderConfig:
    N: int
    cp_len: int
    DATA_BINS: np.ndarray
    clockwise: bool = False

    distance_power: float = 1.0
    distance_eps: float = 1.0

    pll_cfg: "PLLConfig" = field(default_factory=PLLConfig)
    sigma_cfg: "SigmaTrackerConfig" = field(default_factory=SigmaTrackerConfig)
    drift_guard: "DriftGuardConfig" = field(default_factory=DriftGuardConfig)


# ========== 每子载波噪声跟踪器 ==========
class SigmaTracker:
    """
    逐符号跟踪噪声 σ（实/虚），EMA：σ_t = (1-α_t)*σ_{t-1} + α_t*σ_inst。
    α_t 由 SNR(dB) 压缩后的置信度给出，范围 [alpha_min, alpha_max]。
    """
    def __init__(self, Nd: int, cfg: SigmaTrackerConfig):
        self.cfg = cfg
        self.Nd = Nd
        if cfg.per_sc:
            self.sigma_r = np.full((Nd,), cfg.init_sigma, dtype=float)
            self.sigma_i = np.full((Nd,), cfg.init_sigma, dtype=float)
        else:
            self.sigma_r = np.array([cfg.init_sigma], dtype=float)
            self.sigma_i = np.array([cfg.init_sigma], dtype=float)

    def _conf(self, snr_db: float) -> float:
        # 把 SNR(dB) 压到 [0,1]
        return float(1.0 / (1.0 + np.exp(-(snr_db - self.cfg.snr_mid_db) / max(1e-6, self.cfg.snr_scale))))

    def update(self, const_pll: np.ndarray, snr_med_db: float) -> Dict[str, np.ndarray]:
        s = np.asarray(const_pll).reshape(-1)  # [Nd]
        hard = _qpsk_hard(s)
        err = s - hard

        if self.cfg.per_sc:
            # 轻量：|err| → σ 的比例近似（高效、足够鲁棒）
            sigma_inst_r = np.abs(err.real) * 1.2533
            sigma_inst_i = np.abs(err.imag) * 1.2533
        else:
            med_r = np.median(err.real); mad_r = 1.4826 * np.median(np.abs(err.real - med_r))
            med_i = np.median(err.imag); mad_i = 1.4826 * np.median(np.abs(err.imag - med_i))
            sigma_inst_r = np.full_like(self.sigma_r, max(mad_r, 1e-6), dtype=float)
            sigma_inst_i = np.full_like(self.sigma_i, max(mad_i, 1e-6), dtype=float)

        conf = self._conf(float(snr_med_db))
        a = float(np.clip(self.cfg.alpha_min + (self.cfg.alpha_max - self.cfg.alpha_min) * conf,
                          self.cfg.alpha_min, self.cfg.alpha_max))
        self.sigma_r = (1.0 - a) * self.sigma_r + a * sigma_inst_r
        self.sigma_i = (1.0 - a) * self.sigma_i + a * sigma_inst_i
        return {"sigma_r": self.sigma_r.copy(), "sigma_i": self.sigma_i.copy()}


class DD_CPE_PLL:
    """
    判决导向单环 CPE PLL（自适应步长/门限）。
    新接口：DD_CPE_PLL(cfg: PLLConfig)
    兼容旧接口：DD_CPE_PLL(alpha=..., alpha_min=..., ...) —— 会内部转成 cfg。
    """
    def __init__(self, cfg: Optional[PLLConfig] = None, **legacy_kwargs):
        if cfg is None:
            # 兼容旧用法：把散参打包成配置
            cfg = PLLConfig(**legacy_kwargs)
        self.cfg = cfg

        self.theta = 0.0
        self.ema_e = 0.0
        self.ema_snr_db = cfg.snr_th_db

    @staticmethod
    def _wrap_pi(x: float) -> float:
        return (x + np.pi) % (2*np.pi) - np.pi

    def _confidence(self, snr_db: float, e_abs: float) -> float:
        c_snr = 1.0 / (1.0 + np.exp(-(snr_db - self.cfg.snr_mid_db) / max(1e-6, self.cfg.snr_scale)))
        c_err = max(0.0, 1.0 - (e_abs / (np.pi/2.0)))
        return float(np.clip(c_snr * c_err, 0.0, 1.0))

    def step(self, const, snr_med_db: float, hard=None):
        s = np.asarray(const).ravel()
        if hard is None:
            # 依赖你已有的 _qpsk_hard
            from .batch import _qpsk_hard
            hard = _qpsk_hard(s)

        e = np.angle(np.vdot(hard, s * np.exp(-1j * self.theta)))
        e_abs = abs(e)

        # EMA
        b = self.cfg.beta
        self.ema_e = b * self.ema_e + (1 - b) * e_abs
        self.ema_snr_db = b * self.ema_snr_db + (1 - b) * float(snr_med_db)

        # 自适应
        conf = self._confidence(self.ema_snr_db, self.ema_e)
        a_t = np.clip(self.cfg.alpha_min + (self.cfg.alpha_max - self.cfg.alpha_min) * conf,
                      self.cfg.alpha_min, self.cfg.alpha_max)
        th_t = np.clip(self.cfg.snr_th_min_db + (self.cfg.snr_th_max_db - self.cfg.snr_th_min_db) * (1.0 - conf),
                       self.cfg.snr_th_min_db, self.cfg.snr_th_max_db)

        if (snr_med_db >= th_t) and (e_abs < np.pi/2):
            self.theta = self._wrap_pi(self.theta + a_t * e)
        if (snr_med_db >= th_t) and (e_abs < np.pi / 2):
            return s * np.exp(-1j * e)
        else:
            return s * np.exp(-1j * self.theta)


# ========== 主类：段构建 + 质量评估 + 逐符号追踪 ==========
class OFDMSoftDecoder:
    """
    使用方式：
      1) dec = OFDMSoftDecoder(cfg)
      2) dec.init_from_preamble(pilots_td, pilot_ref_fd)  # 前导初始化 origin_Hf / delta / phi / σ
      3) out = dec.process_sequence(symbols_td, comb_pilot_pos, data_pos, pilot_ref_comb_fd)

    处理完成后，可通过：
      - out['const']        : [n_data, Nd]  （PLL 后星座）
      - out['snr_med_db']   : [n_data]
      - out['sigma_r/i']    : [n_data, Nd]（或 [n_data,1] 若 per_sc=False）
      - out['data_idx']     : [n_data]      （对应的绝对 OFDM 索引）
      - out['comb_q']       : [n_comb]
      - out['seg_params']   : 每段的 (start_idx, end_idx, delta, phi_step, q)

    并可读取/修改超参：
      - dec.get_hparams() / dec.set_hparams(...)
      - 直接访问 dec.pll / dec.sigma_trk.cfg（例如动态调参）
    """

    def __init__(self, cfg: DecoderConfig):
        self.cfg = cfg
        self.Nd = cfg.DATA_BINS.size
        self.symbol_len = cfg.N + cfg.cp_len

        self.pll = DD_CPE_PLL(cfg.pll_cfg)
        self.sigma_trk = SigmaTracker(self.Nd, cfg.sigma_cfg)

        self._delta_hist: list[float] = []  # 被“采纳”的 delta（已过守护）
        self._phi_hist: list[float] = []
        self._idx_hist: list[int] = []

        self._delta_raw_hist: list[float] = []  # comb 每次“原始拟合”的 delta/phi
        self._phi_raw_hist: list[float] = []
        self._idx_raw_hist: list[int] = []

        self._guard_events: list[dict] = []  # 每次守护的详细信息
        self._warn_callback = None  # 可选：用户自定义告警回调

    # ---------- 前导初始化 ----------
    def init_from_preamble(self, pilots_td: np.ndarray, pilot_ref_fd: np.ndarray):
        """
        pilots_td: [num_pilot, N]（已去 CP）
        pilot_ref_fd: [num_pilot, N]
        """
        Hf_pilot = evaluate_H_f(pilots_td, pilots_fd=pilot_ref_fd)  # [num_pilot, N]
        # 你已有的全局估计接口（保持一致）
        from .batch import estimate_drift_and_origin
        d0, p0, origin = estimate_drift_and_origin(Hf_pilot, N=self.cfg.N, symbol_len=self.symbol_len)
        self.origin_Hf = Hf_pilot[-1]
        self.delta = float(d0)
        self.phi_step = float(p0)

        # 用最后一个前导初始化 σ（保守）
        Xf = np.fft.fft(pilots_td[-1]) / Hf_pilot[-1]
        Xd = Xf[self.cfg.DATA_BINS]
        hard = _qpsk_hard(Xd)
        err = Xd - hard
        sig0 = max(1.4826 * np.median(np.abs(err.real)), 1e-3)
        self.sigma_trk.sigma_r[...] = sig0
        self.sigma_trk.sigma_i[...] = sig0
        self._register_drift(self.delta, self.phi_step, idx=-1)


    # ---------- 外部可读/可改的超参接口 ----------
    def get_hparams(self):
        return {
            "pll": asdict(self.cfg.pll_cfg),
            "sigma": asdict(self.sigma_trk.cfg),
            "distance_power": self.cfg.distance_power,
            "distance_eps": self.cfg.distance_eps,
        }

    def set_hparams(self, **kwargs):
        # 允许直接传 pll_cfg 字段
        for k, v in list(kwargs.items()):
            if hasattr(self.cfg.pll_cfg, k):
                setattr(self.cfg.pll_cfg, k, type(getattr(self.cfg.pll_cfg, k))(v))
                setattr(self.pll, k, type(getattr(self.pll, k))(v))  # 同步到环对象（通过属性桥接）
                kwargs.pop(k, None)

        # SigmaTrackerConfig
        for k, v in list(kwargs.items()):
            if hasattr(self.sigma_trk.cfg, k):
                setattr(self.sigma_trk.cfg, k, type(getattr(self.sigma_trk.cfg, k))(v))
                setattr(self.sigma_trk, k, type(getattr(self.sigma_trk, k))(v))
                kwargs.pop(k, None)

        # 段权重
        if "distance_power" in kwargs: self.cfg.distance_power = float(kwargs.pop("distance_power"))
        if "distance_eps"   in kwargs: self.cfg.distance_eps   = float(kwargs.pop("distance_eps"))

    # ---------- 序列处理（核心）：段构建 + 质量评估 + 逐符号追踪 ----------
    def process_sequence(self,
                         symbols_td: np.ndarray,            # [M, N] 时域 OFDM（数据+comb，已去 CP）
                         comb_pilot_pos: np.ndarray,         # [n_comb] comb 的绝对索引（0..M-1）
                         data_pos: np.ndarray,               # [n_data] data 的绝对索引（0..M-1）
                         pilot_ref_comb_fd: np.ndarray       # [n_comb, N] comb 的频域参考
                         ) -> Dict[str, np.ndarray]:
        """
        内部步骤：
          - 以“前导最后一块”为段起点参考 H_start，遇 comb 就结算上一段：
            * 用 H_start 与 H_comb 拟合跨段 (delta, phi_step)
            * 用非同源外推的 H_pred 评估 comb 质量 q
            * 对上一段所有 data：构造 (w1,w2)，H_used = w1*H1 + w2*H2，逐符号 PLL 与 σ 跟踪
          - 刷新段起点参考为该 comb，进入下一段；末段没有 comb 则退化 w1=1,w2≈0

        返回：
          - const        : [n_data, Nd]  PLL 后星座（便于你后续 MMSE/LLR）
          - snr_med_db   : [n_data]
          - sigma_r/i    : [n_data, Nd]（或 [n_data,1]）
          - data_idx     : [n_data]      绝对 OFDM 索引
          - comb_q       : [n_comb]
          - seg_params   : [n_seg, 5]    每段 (start_idx, end_idx, delta, phi_step, q_used)
        """
        assert self.origin_Hf is not None, "Call init_from_preamble() first."

        DATA_BINS = self.cfg.DATA_BINS
        M = symbols_td.shape[0]
        comb_pilot_pos = np.asarray(comb_pilot_pos, dtype=int)
        data_pos = np.asarray(data_pos, dtype=int)

        # 输出累积器
        const_list: List[np.ndarray] = []
        snr_list: List[float] = []
        sigma_r_list: List[np.ndarray] = []
        sigma_i_list: List[np.ndarray] = []
        data_idx_list: List[int] = []
        comb_q_list: List[float] = []
        seg_meta: List[Tuple[int,int,float,float,float]] = []

        # 段状态
        seg_start_idx = -1
        seg_start_H   = self.origin_Hf.copy()
        seg_buf: List[Tuple[int, np.ndarray]] = []   # (i_abs, sym_td)
        j_comb = 0
        have_last_fit = False
        last_d = float(self.delta)
        last_p = float(self.phi_step)

        # 遍历整个序列
        for i_abs in range(M):
            is_comb = (j_comb < comb_pilot_pos.size) and (i_abs == comb_pilot_pos[j_comb])
            if not is_comb:
                # data：先缓存在本段
                if i_abs in set(data_pos):
                    seg_buf.append((i_abs, symbols_td[i_abs]))
                continue

            # === comb 到达：估计 H_comb、拟合漂移、非同源质量 q ===
            sym_comb_td = symbols_td[i_abs]
            H_comb = evaluate_H_f(sym_comb_td, pilots_fd=pilot_ref_comb_fd[j_comb])  # [N]
            gap = int(i_abs - seg_start_idx)
            d_raw, p_raw = _fit_drift_between(seg_start_H, H_comb, gap, N=self.cfg.N, symbol_len=self.symbol_len, return_phi=True)
            self._record_raw_drift(d_raw, p_raw, idx=i_abs)
            # 守护 + 可能替换
            d_j, p_j, guard_info = self._guard_drift(d_raw, p_raw, idx=i_abs)
            # 若发生替换，打印/回调
            if guard_info.get("guarded") and self.cfg.drift_guard.warn_on_replace:
                self._emit_guard_warning(guard_info, idx=i_abs)

            # 非同源 H_pred（从段起点外推到 comb）
            H_pred = correct_H_f(seg_start_H, delta=d_j, N=self.cfg.N, index=gap, symbol_len=self.symbol_len,
                                 fixed_phase_shift_factor=p_j)
            Xd = get_constellation(symbols_td=symbols_td, H_used=H_pred, DATA_BINS=DATA_BINS)
            Rd = pilot_ref_comb_fd[j_comb][DATA_BINS]
            snr_sc = np.abs(Rd)**2 / (np.abs(Xd - Rd)**2 + 1e-12)
            snr_db_med_q = float(10.0 * np.log10(np.median(np.clip(snr_sc, 1e-12, None))))
            q_j = float(1.0 / (1.0 + np.exp(-(snr_db_med_q - 5.0) / 2.0)))
            comb_q_list.append(q_j)

            # === 处理上一段所有 data ===
            near_idx = i_abs
            for (i_data_abs, sym_td) in seg_buf:
                # 距离 & 权重
                dt1 = float(i_data_abs - seg_start_idx)
                dt2 = float(i_data_abs - near_idx)
                d1 = max(dt1, 0.0)
                d2 = abs(dt2)
                w1 = 1.0 / np.power(d1 + self.cfg.distance_eps, self.cfg.distance_power)
                w2 = q_j * (1.0 / np.power(d2 + self.cfg.distance_eps, self.cfg.distance_power))
                s = w1 + w2
                w1 /= s; w2 /= s

                # 两路外推并加权
                H1 = correct_H_f(seg_start_H, delta=d_j, N=self.cfg.N, index=dt1, symbol_len=self.symbol_len,
                                 fixed_phase_shift_factor=p_j)
                H2 = correct_H_f(H_comb,     delta=d_j, N=self.cfg.N, index=dt2, symbol_len=self.symbol_len,
                                 fixed_phase_shift_factor=p_j)
                H_used = w1 * H1 + w2 * H2

                # 均衡 -> 逐符号 PLL -> σ 跟踪 -> MMSE收缩
                const_zf = get_constellation(sym_td, H_used, DATA_BINS=DATA_BINS)
                snr_med = float(pll_snr_median(const_zf))
                const_pll = self.pll.step(const_zf, snr_med)
                sigs = self.sigma_trk.update(const_pll, snr_med)
                # sigs = robust_sigma(const_pll)
                Habs2_i = np.abs(H_used[self.cfg.DATA_BINS]) ** 2
                const_mmse = mmse_shrinkage(const_pll, Habs2_i, sigs)

                # 记录
                const_list.append(const_mmse)  # ← 只保存“MMSE 后”的星座，建议 copy 更安全
                snr_list.append(snr_med)
                sigma_r_list.append(sigs["sigma_r"].copy())
                sigma_i_list.append(sigs["sigma_i"].copy())
                data_idx_list.append(i_data_abs)

            if seg_buf:
                seg_meta.append((seg_start_idx, near_idx, float(d_j), float(p_j), float(q_j)))
            seg_buf.clear()
            self._register_drift(d_j, p_j, idx=i_abs)

            # === 调整追踪器超参（越好越激进） ===
            self._tune_from_q(q_j, snr_db_med_q)

            # === 刷新段起点为当前 comb，准备下一段 ===
            seg_start_idx = i_abs
            seg_start_H   = H_comb.copy()
            last_d, last_p = float(d_j), float(p_j)
            have_last_fit = True
            j_comb += 1

        # === 末段（无右端 comb） ===
        if seg_buf:
            for (i_data_abs, sym_td) in seg_buf:
                dt1 = float(i_data_abs - seg_start_idx)
                d_j = last_d if have_last_fit else float(self.delta)
                p_j = last_p if have_last_fit else float(self.phi_step)
                H1 = correct_H_f(seg_start_H, delta=d_j, N=self.cfg.N, index=dt1, symbol_len=self.symbol_len,
                                 fixed_phase_shift_factor=p_j)
                H_used = H1  # w2≈0 的退化
                const_zf = get_constellation(sym_td, H_used, DATA_BINS=DATA_BINS)
                snr_med = float(pll_snr_median(const_zf))
                const_pll = self.pll.step(const_zf, snr_med)
                sigs = self.sigma_trk.update(const_pll, snr_med)
                # sigs = robust_sigma(const_pll)
                Habs2_i = np.abs(H_used[self.cfg.DATA_BINS]) ** 2
                const_mmse = mmse_shrinkage(const_pll, Habs2_i, sigs)

                const_list.append(const_mmse)  # ← 只保存 MMSE 后的星座
                snr_list.append(snr_med)
                sigma_r_list.append(sigs["sigma_r"].copy())
                sigma_i_list.append(sigs["sigma_i"].copy())
                data_idx_list.append(i_data_abs)
            seg_meta.append((seg_start_idx, int(M), float(d_j), float(p_j), 1.0))  # 尾段 q=1.0 作为占位

        # 按 OFDM 索引从小到大排序输出（以防 data_pos 非升序）
        order = np.argsort(np.asarray(data_idx_list))
        const_arr  = np.stack(const_list, axis=0)[order]
        snr_arr    = np.asarray(snr_list, dtype=float)[order]
        sigma_r_arr= np.stack(sigma_r_list, axis=0)[order]
        sigma_i_arr= np.stack(sigma_i_list, axis=0)[order]
        data_idx   = np.asarray(data_idx_list, dtype=int)[order]

        return {
            "const": const_arr,                 # [n_data, Nd]
            "snr_med_db": snr_arr,             # [n_data]
            "sigma_r": sigma_r_arr,            # [n_data, Nd] 或 [n_data,1]
            "sigma_i": sigma_i_arr,            # [n_data, Nd] 或 [n_data,1]
            "data_idx": data_idx,              # [n_data]
            "comb_q": np.asarray(comb_q_list, dtype=float),     # [n_comb]
            "seg_params": np.asarray(seg_meta, dtype=float),    # [n_seg, 5]: start, end, delta, phi_step, q
        }

    # —— 内部：根据 q 调整追踪器学习率区间（质量越好越激进） —— #
    def _tune_from_q(self, q: float, snr_db: Optional[float] = None):
        q = float(np.clip(q, 0.0, 1.0))
        def L(lo, hi): return float(lo + (hi - lo) * q)

        # PLL
        self.pll.cfg.alpha_min     = L(0.03, 0.10)
        self.pll.cfg.alpha_max     = L(0.20, 0.40)
        self.pll.cfg.snr_th_min_db = L(4.0,  2.0)
        self.pll.cfg.snr_th_max_db = L(10.0, 8.0)
        if snr_db is not None:
            self.pll.cfg.snr_mid_db = 0.7 * self.pll.cfg.snr_mid_db + 0.3 * float(snr_db)

        # σ 跟踪
        self.sigma_trk.cfg.alpha_min = L(0.03, 0.08)
        self.sigma_trk.cfg.alpha_max = L(0.15, 0.30)

    def _register_drift(self, delta: float, phi_step: float, idx: int):
        """记录一次被采用的漂移参数；控制历史长度。"""
        self._delta_hist.append(float(delta))
        self._phi_hist.append(float(phi_step))
        self._idx_hist.append(int(idx))

        # 截断历史长度
        hm = self.cfg.drift_guard.hist_max
        if len(self._delta_hist) > hm:
            self._delta_hist = self._delta_hist[-hm:]
            self._phi_hist = self._phi_hist[-hm:]
            self._idx_hist = self._idx_hist[-hm:]

    @staticmethod
    def _mean_std(arr: np.ndarray) -> tuple[float, float]:
        """返回均值与样本标准差；长度<2 时 std 给个很小的正数避免除零。"""
        if arr.size < 2:
            return (float(arr.mean()) if arr.size else 0.0, 1e-12)
        mu = float(arr.mean())
        std = float(arr.std(ddof=1))
        return (mu, max(std, 1e-12))

    def _linear_extrapolate(self, idx_hist: np.ndarray, val_hist: np.ndarray, predict_idx: int) -> float:
        """
        用最近 K 个样本做一次线性外推；不足 2 个样本时返回最后一个。
        """
        n = int(val_hist.size)
        if n == 0:
            return 0.0
        if n == 1:
            return float(val_hist[-1])

        # 选最近 K 个点（K 至少 2）
        K = min(n, max(2, int(self.cfg.drift_guard.fit_window)))
        x = idx_hist[-K:].astype(float)
        y = val_hist[-K:].astype(float)
        try:
            k, b = np.polyfit(x, y, deg=1)
            return float(k * float(predict_idx) + b)
        except Exception:
            return float(val_hist[-1])

    def _guard_drift(self, d_new: float, p_new: float, idx: int) -> tuple[float, float, dict]:
        """
        均值±nσ 检测漂移异常；异常时用“线性外推”替代。
        返回 (d_used, p_used, info)
        """
        if not self.cfg.drift_guard.enabled or len(self._delta_hist) < 2:
            return float(d_new), float(p_new), {"guarded": False, "mode": "none"}

        n_sigma = float(self.cfg.drift_guard.n_sigma)
        K = min(len(self._delta_hist), max(3, self.cfg.drift_guard.fit_window))

        d_hist = np.asarray(self._delta_hist[-K:], dtype=float)
        p_hist = np.asarray(self._phi_hist[-K:], dtype=float)
        i_hist = np.asarray(self._idx_hist[-K:], dtype=int)

        d_mu, d_std = self._mean_std(d_hist)
        p_mu, p_std = self._mean_std(p_hist)

        d_lo, d_hi = d_mu - n_sigma * d_std, d_mu + n_sigma * d_std
        p_lo, p_hi = p_mu - n_sigma * p_std, p_mu + n_sigma * p_std

        d_ok = (d_lo <= d_new <= d_hi)
        p_ok = (p_lo <= p_new <= p_hi)

        info = {
            "guarded": (not d_ok) or (not p_ok),
            "d_new": float(d_new), "p_new": float(p_new),
            "d_mu": d_mu, "d_std": d_std, "p_mu": p_mu, "p_std": p_std,
            "d_bounds": (d_lo, d_hi), "p_bounds": (p_lo, p_hi),
        }

        d_used = float(d_new)
        p_used = float(p_new)

        if not d_ok:
            d_hat = self._linear_extrapolate(i_hist, d_hist, predict_idx=idx)
            d_used = d_hat
            info["d_replaced_with"] = d_hat
        if not p_ok:
            p_hat = self._linear_extrapolate(i_hist, p_hist, predict_idx=idx)
            p_used = p_hat
            info["p_replaced_with"] = p_hat

        return d_used, p_used, info


    def _record_raw_drift(self, d_raw: float, p_raw: float, idx: int):
        self._delta_raw_hist.append(float(d_raw))
        self._phi_raw_hist.append(float(p_raw))
        self._idx_raw_hist.append(int(idx))
        # 控制长度
        hm = self.cfg.drift_guard.hist_max
        if len(self._delta_raw_hist) > hm:
            self._delta_raw_hist = self._delta_raw_hist[-hm:]
            self._phi_raw_hist   = self._phi_raw_hist[-hm:]
            self._idx_raw_hist   = self._idx_raw_hist[-hm:]

    def _emit_guard_warning(self, info: dict, idx: int):
        # 组织人类可读信息
        pre = self.cfg.drift_guard.warn_prefix
        parts = [f"{pre} idx={idx}"]
        if "d_replaced_with" in info:
            dlo, dhi = info["d_bounds"]
            parts.append(
                f"delta raw={info['d_new']:.6g} out of [{dlo:.6g},{dhi:.6g}] → used {info['d_replaced_with']:.6g}"
            )
        if "p_replaced_with" in info:
            plo, phi = info["p_bounds"]
            parts.append(
                f"phi   raw={info['p_new']:.6g} out of [{plo:.6g},{phi:.6g}] → used {info['p_replaced_with']:.6g}"
            )
        msg = " | ".join(parts)

        # 打印 & 回调
        print(msg)
        self._guard_events.append({"idx": idx, **info, "message": msg})
        if callable(self._warn_callback):
            try:
                self._warn_callback(msg, info)
            except Exception:
                pass

    def get_drift_history(self, kind: str = "accepted") -> dict:
        """
        kind in {"accepted","raw"}：
          - accepted：返回守护后“被采纳”的 delta/phi 与对应 idx
          - raw     ：返回 comb 处原始拟合的 delta/phi 与对应 idx
        返回：{"idx": np.ndarray, "delta": np.ndarray, "phi": np.ndarray}
        """
        if kind == "accepted":
            idx = np.asarray(self._idx_hist, dtype=int)
            d = np.asarray(self._delta_hist, dtype=float)
            p = np.asarray(self._phi_hist, dtype=float)
        elif kind == "raw":
            idx = np.asarray(self._idx_raw_hist, dtype=int)
            d = np.asarray(self._delta_raw_hist, dtype=float)
            p = np.asarray(self._phi_raw_hist, dtype=float)
        else:
            raise ValueError("kind must be 'accepted' or 'raw'")
        return {"idx": idx, "delta": d, "phi": p}

    def get_guard_events(self) -> list[dict]:
        """返回每次发生替换时的详细记录（含 message）。"""
        return list(self._guard_events)