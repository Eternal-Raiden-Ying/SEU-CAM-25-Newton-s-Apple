import numpy as np
import numpy.fft
import sounddevice as sd


# TODO: func:
#       constellation reflection (more than QPSK)
#       X_f_est approximation / calibration
#       check the func get bytes

def get_symbols(record:np.ndarray, cp_len, N, **kwargs) -> np.ndarray:
    assert record.ndim == 1, f"Unexpected dimension of record, with shape{record.shape}"
    symbol_len = N + cp_len
    n_symbols = record.size // symbol_len
    ofdm_symbols = np.array([
        record[i * symbol_len + cp_len: (i + 1) * symbol_len]
        for i in range(n_symbols)
    ])
    return ofdm_symbols


def get_constellation(symbols: np.ndarray, H_f: np.ndarray, approximation, **kwargs):
    """
    return constellation with meanings, exclude the conjugate part
    :param symbols:
    :param H_f:
    :param approximation:
    :param kwargs:
    :return:
    """
    if 'symbol_len' in kwargs:
        # when more than 1 symbol is given, symbol length should be specified
        symbol_len = kwargs['symbol_len']
        symbols = symbols.reshape(-1, symbol_len)
        Y_f = np.fft.fft(symbols, axis=1)
    else:
        assert symbols.ndim == 1, "when more than 1 symbol is given, symbol length should be specified"
        symbol_len = symbols.size
        Y_f = np.fft.fft(symbols)

    data_bins = np.arange(1, symbol_len // 2)
    X_f = Y_f / H_f
    data_carriers = X_f[data_bins] if symbols.ndim == 1 else X_f[:, data_bins]

    # TODO: error repairing here or in the approximation
    return approximation(data_carriers)


def simple_approximate(data: np.ndarray):
    # TODO: remains to be completed
    #       here use Quadrant to judge, so normalization is not needed
    real = np.real(data)
    imag = np.imag(data)
    approximate = np.zeros_like(data)
    approximate[real >= 0] += 1
    approximate[real < 0] -= 1
    approximate[imag > 0] += 1j
    approximate[imag <= 0] -= 1j
    return approximate


def non_approximate(data: np.ndarray):
    return data

def QPSK_reflection(data: np.ndarray, *, clockwise:bool = False, judge_radius = 0.5):
    data_dim = data.ndim
    res = np.stack([np.zeros_like(data), np.zeros_like(data)], axis=data_dim)
    if clockwise:
        res[np.where(np.abs(data-(1+1j)) < judge_radius)] = [0,0]
        res[np.where(np.abs(data-(-1+1j)) < judge_radius)] = [1,0]
        res[np.where(np.abs(data-(-1-1j)) < judge_radius)] = [1,1]
        res[np.where(np.abs(data-(1-1j)) < judge_radius)] = [0,1]
    else:
        res[np.where(np.abs(data-(1+1j)) < judge_radius)] = [0,0]
        res[np.where(np.abs(data-(-1+1j)) < judge_radius)] = [0,1]
        res[np.where(np.abs(data-(-1-1j)) < judge_radius)] = [1,1]
        res[np.where(np.abs(data-(1-1j)) < judge_radius)] = [1,0]

    res = np.real(res)
    res = res.astype(int)
    return res

def get_bytes(binary_data: np.ndarray, bitorder='big'):
    if bitorder not in ['big', 'little']:
        raise ValueError("bitorder should in ['big', 'little']")
    bits = binary_data.flatten()[:binary_data.size // 8 * 8].reshape(-1, 8)
    try:
        res = np.packbits(bits, axis=-1, bitorder=bitorder)
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
    test_arr = np.array([[1+1j,1-1j, 1-1j],[-1+1j,-1-1j, 1-1j],[-1+1j,-1-1j, 1-1j]])
    print(QPSK_reflection(test_arr))
