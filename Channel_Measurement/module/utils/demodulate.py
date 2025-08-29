import numpy as np
import numpy.fft
import sounddevice as sd


# TODO: func:
#       constellation reflection (more than QPSK)
#       X_f_est approximation / calibration
#       check the func get bytes


def simple_approximate(data: np.ndarray):
    """
        simply make a hard decision (based on the constellation's quadrant)
    :param data: constellations received
    :return: approximate constellations
    """
    # TODO: remains to be completed
    #       here use Quadrant to judge, so normalization is not needed
    real = np.real(data)
    imag = np.imag(data)
    approximate = np.zeros_like(data)
    approximate[real >= 0] += 1
    approximate[real < 0] -= 1
    approximate[imag > 0] += 1j
    approximate[imag <= 0] -= 1j
    return approximate / np.sqrt(2)


def non_approximate(data: np.ndarray):
    """
        Identity, do nothing
    :param data:
    :return:
    """
    return data


def normalize_approximate(data:np.ndarray):
    eps = 1e-8
    return data / (np.abs(data)+eps)

def get_symbols(record:np.ndarray, cp_len, N, **kwargs) -> np.ndarray:
    """
        from record get symbols (without cyclic prefix)
        just make a shape change and drop the cp part,
        for a sequence shorter than symbol_len (cp_len+N), it will be dropped
    :param record: record in TD
    :param cp_len:
    :param N:
    :param kwargs: for further develop
    :return:
    """
    assert record.ndim == 1, f"Unexpected dimension of record, with shape{record.shape}"
    symbol_len = N + cp_len
    n_symbols = record.size // symbol_len
    ofdm_symbols = np.array([
        record[i * symbol_len + cp_len: (i + 1) * symbol_len]
        for i in range(n_symbols)
    ])
    return ofdm_symbols


def get_constellation(symbols_td: np.ndarray, H_used: np.ndarray, *, DATA_BINS: np.ndarray) -> np.ndarray:
    """
    统一版等化：支持 [N]/[ns,N]。
    返回：若输入为 [N] -> [Nd]；若 [ns,N] -> [ns,Nd]
    """
    Yf = np.fft.fft(symbols_td, axis=-1)
    Xf = Yf / H_used
    return Xf[..., DATA_BINS]


# ========= QPSK 基础 =========
def _qpsk_hard(z: np.ndarray) -> np.ndarray:
    """
    把星座点硬判到最近标准 QPSK 点 (±1 ± j)/√2；保持输入形状。
    """
    z = np.asarray(z)
    re = np.sign(z.real)
    im = np.sign(z.imag)
    re[re == 0] = 1
    im[im == 0] = 1
    return (re + 1j * im) / np.sqrt(2)

def QPSK_reflection(z: np.ndarray, *, clockwise: bool = False) -> np.ndarray:
    """
    constellation -> bits（**不是**就近映射）
    - 输入:  z 复数星座，[Nd] 或 [n, Nd]
    - 输出:  uint8 比特数组；若输入 [Nd] -> [Nd*2]；若 [n,Nd] -> [n, Nd*2]
    规则：
      (0,0) -> 第一象限（I>0,Q>0） 恒定
      (0,1) -> clockwise=False 时对应第二象限；clockwise=True 时对应第四象限
    实现（向量化）：
      设 sI = (Re(z) >= 0), sQ = (Im(z) >= 0)
      clockwise=True :  bI = ~sI, bQ = ~sQ
      clockwise=False:  bI = ~sQ, bQ = ~sI
    """
    z = np.asarray(z)
    if z.ndim == 1:
        sI = (z.real >= 0)
        sQ = (z.imag >= 0)
        if clockwise:
            bI = (~sI).astype(np.uint8)
            bQ = (~sQ).astype(np.uint8)
        else:
            bI = (~sQ).astype(np.uint8)
            bQ = (~sI).astype(np.uint8)
        bits = np.stack([bI, bQ], axis=-1).reshape(-1)
        return bits
    elif z.ndim == 2:
        sI = (z.real >= 0)
        sQ = (z.imag >= 0)
        if clockwise:
            bI = (~sI).astype(np.uint8)
            bQ = (~sQ).astype(np.uint8)
        else:
            bI = (~sQ).astype(np.uint8)
            bQ = (~sI).astype(np.uint8)
        bits2 = np.stack([bI, bQ], axis=-1)           # [n, Nd, 2]
        return bits2.reshape(z.shape[0], -1)          # [n, Nd*2]
    else:
        raise ValueError("QPSK_reflection: 仅支持 1D 或 2D 输入")

def get_bytes(binary_data: np.ndarray, bitorder='big'):
    """
        turn a binary np.ndarray into a sequence of bytes
        raise an error when given data is not binary
        NOTE: func will drop the excessive bit, info given yet
    :param binary_data: binary data
    :param bitorder: bit order, big default, intuitive
    :return: bytes sequence
    """
    if bitorder not in ['big', 'little']:
        raise ValueError("bitorder should in ['big', 'little']")
    bits = binary_data.flatten()[:binary_data.size // 8 * 8]
    if binary_data.size > bits.size:
        print(f"{binary_data.size - bits.size} bits were dropped, see details at get_bytes()")
    try:
        res = np.packbits(bits, bitorder=bitorder)
        return res
    except:
        print("Error occurred when get_bytes is invoked, make sure the given data is binary")
        raise RuntimeError


def evaluate_H_f(known_symbols: np.ndarray, pilot_signals: np.ndarray):
    """
        evaluate H(f) by pilot signals (when several is given, H is calculated through average)
    :param known_symbols: symbol extracted from received pilot signal (exclude CP)
    :param pilot_signals: original pilot signal (include the conjugate part)
    :return:
    """

    Y_f = np.fft.fft(known_symbols, axis=-1)
    H_f = Y_f/pilot_signals if Y_f.ndim == 1 else (Y_f/pilot_signals).mean(axis=0)
    return H_f





# here is for unit test
if __name__ == "__main__":
    test_arr = np.array([[1+1j,1-1j, 1-1j],[-1+1j,-1-1j, 1-1j],[-1+1j,-1-1j, 1-1j]])/np.sqrt(2)
    print(QPSK_reflection(test_arr))
