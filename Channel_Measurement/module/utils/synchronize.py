import numpy as np
from scipy.signal import correlate

# ========= 同步 =========
def synchronize(rx: np.ndarray, chirp_template: np.ndarray, mode: str = "full"):
    """
    用 scipy.signal.correlate 做时域同步。
    返回:
      ofdm_start : int  (OFDM 起始点 = argmax(|corr|) + 1)
      corr       : np.ndarray  (相关序列)
      chirp_start: int  (chirp 起点)
    """
    corr = correlate(rx, chirp_template, mode=mode)
    ofdm_start = int(np.argmax(np.abs(corr))) + 1
    chirp_start = ofdm_start - chirp_template.size
    return ofdm_start, corr, chirp_start