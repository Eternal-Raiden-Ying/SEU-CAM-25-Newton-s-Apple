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


def get_constellation(symbols: np.ndarray, H_f: np.ndarray, approximation=non_approximate, **kwargs):
    """
        return constellation with meanings, exclude the conjugate part
    :param symbols: symbols in TD (without cp), you can get it from get_symbols()
    :param H_f: estimated H(f), you can get it from evaluate_H_f()
    :param approximation: choose a approximation rule, default non approximate
    :param kwargs: for further develop
        'symbol_len': when more than 1 symbol is given, symbol length should be specified
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


def QPSK_reflection(data: np.ndarray, *, clockwise:bool = False, judge_radius=0.5):
    """
        constellations -->  binary np.ndarray
        better use with approximation (even simple)
        # TODO: reamin to turn it to be a BEC
    :param data: constellations
    :param clockwise: details at QPSK_mapping(), default anticlockwise
    :param judge_radius:
    :return:
    """
    data_dim = data.ndim
    res = np.stack([np.ones_like(data), np.ones_like(data)], axis=data_dim) * (-1)
    if clockwise:
        res[np.where(np.abs(data-(1+1j)/np.sqrt(2)) < judge_radius)] = [0,0]
        res[np.where(np.abs(data-(-1+1j)/np.sqrt(2)) < judge_radius)] = [1,0]
        res[np.where(np.abs(data-(-1-1j)/np.sqrt(2)) < judge_radius)] = [1,1]
        res[np.where(np.abs(data-(1-1j)/np.sqrt(2)) < judge_radius)] = [0,1]
    else:
        res[np.where(np.abs(data-(1+1j)/np.sqrt(2)) < judge_radius)] = [0,0]
        res[np.where(np.abs(data-(-1+1j)/np.sqrt(2)) < judge_radius)] = [0,1]
        res[np.where(np.abs(data-(-1-1j)/np.sqrt(2)) < judge_radius)] = [1,1]
        res[np.where(np.abs(data-(1-1j)/np.sqrt(2)) < judge_radius)] = [1,0]

    res = np.real(res)
    res = res.astype(int)
    return res

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
