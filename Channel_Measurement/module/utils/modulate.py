import numpy as np
import sounddevice as sd
import os
from scipy.signal import chirp
from .io_interface import random_bits

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


def generate_chirp(fs, duration, f_l, f_h,*, method='linear'):
    """
        generate a chirp signal in time domain, max val is 1
    :param fs: sampling freq
    :param duration: duration of chirp signal, usually 1 or 2
    :param f_l: low freq (start freq)
    :param f_h: high freq (end freq)
    :param method: ‘linear’, ‘quadratic’, ‘logarithmic’, ‘hyperbolic’, default 'linear'
    :return: a chirp signal, data type of np.ndarray
    """
    t = np.linspace(0, duration, int(fs * duration))
    chirp_sig = chirp(t, f0=f_l, f1=f_h, t1=duration, method='linear')
    return chirp_sig


def QPSK_mapping(data: np.ndarray,*,clockwise=False):
    """
        turn a binary np.ndarray (the code sequence) into a complex sequence,
        without the conjugate part and cp
        remain its shape except the last dim, e.g. (8,2047,2) -> (8,2047)
        output is normalized
    :param data: the binary data, the last dim must be 2, either it would raise Error
    :param clockwise: boolean, anticlockwise default,
        anticlockwise for:          clockwise for:
        (0,1) | (0,0)               (1,0) | (0,0)
        -------------               -------------
        (1,1) | (1,0)               (1,1) | (0,1)
    :return: normalized QPSK constellations
    """
    shape = data.shape
    assert shape[-1] == 2, f"expected length of last dim 2, but received {shape[-1]}"

    dim_order = np.arange(data.ndim).tolist()
    dim_order.pop()
    dim_order.insert(0, data.ndim-1)

    res = np.zeros(shape[:len(shape)-1], dtype=np.complex128)
    bits = np.permute_dims(data, tuple(dim_order))
    # b1b0
    b1 = bits[0]
    b0 = bits[1]
    if clockwise:
        res[np.where(b1 == 0)] += 1
        res[np.where(b1 == 1)] -= 1
        res[np.where(b0 == 0)] += 1j
        res[np.where(b0 == 1)] -= 1j
    else:
        res[np.where(b1 == 0)] += 1j
        res[np.where(b1 == 1)] -= 1j
        res[np.where(b0 == 0)] += 1
        res[np.where(b0 == 1)] -= 1
    return res / np.sqrt(2)


def OFDM_modulate(constellations: np.ndarray, N: int, cp_len: int, *,
                  complement_val=0, padding_zero=False, random_tail=True):
    """
        given constellations and modulate into time signal (with cyclic prefix)
        if use comb-type pilot or block-type pilot, turn to add_pilot() first

        v2: complement_val deprecated, extreme high peak due to continuous identical values,
            I just noticed that, maybe not that reason exactly, but I think constellations
            from random bits is more reasonable

        v3: in order to cooperate with others, using standard below:
            padding zero in 0 and N//2,
            padding random bits to the tail

    :param constellations: just your constellations
    :param N: num of sub carrier waves
    :param cp_len: length of cyclic prefix
    :param complement_val: (deprecated) padding value, pad when constellations.size != k*N, default 0
    :return: time signal with cyclic prefix
    """
    assert constellations.ndim <= 2, "constellations for over 2 dims not supported"
    assert constellations.shape[-1] < N//2, ("in order to have the signal in time domain to be real, "
                                             "length of constellations in each symbol should less than N/2")
    constellation_len = N //2 -1
    num_symbols = int(np.ceil(constellations.size / constellation_len))
    if complement_val:
        print("Warning! complement_val is deprecated, see function OFDM comments for details")
    if num_symbols*constellation_len > constellations.size:
        print("Warning! Better make sure constellations.size is k*(N//2-1), OFDM modulate invoked")
        complement_len = int(num_symbols*constellation_len - constellations.size)
        if random_tail:
            padding_constellations = QPSK_mapping(random_bits(2 * complement_len).reshape(-1, 2))
        else:
            padding_constellations = QPSK_mapping(np.zeros(2 * complement_len).reshape(-1, 2))
        constellations = np.concatenate([constellations.flatten(), padding_constellations])
        constellations = constellations.reshape(num_symbols, constellation_len)
    if padding_zero:
        symbols = np.concatenate([np.zeros((num_symbols,1), dtype=np.int32),
                                  constellations,
                                  np.zeros((num_symbols,1),dtype=np.int32),
                                  np.conjugate(constellations)[:,::-1]],
                                 axis=1
                                 )
    else:
        # padding one
        symbols = np.concatenate([np.ones((num_symbols,1), dtype=np.int32),
                                  constellations,
                                  np.ones((num_symbols,1),dtype=np.int32),
                                  np.conjugate(constellations)[:,::-1]],
                                 axis=1
                                 )

    symbol_td = np.real(np.fft.ifft(symbols, axis=1))
    symbol_with_cp = np.concatenate([symbol_td[:,-cp_len:], symbol_td], axis=1)
    return symbol_with_cp


def add_pilot(data, pilot, data_idx, pilot_idx, *, N=None, padding_clockwise=False):
    """
        NOTE!!! given data and pilot should be 1 dim
        mix the data and pilot with given index,
        this function will check if idx combined is continuous when N is given,
        but only through warning, for there exists situation when data is not enough,
        rest will be padded with constellations from random bits (not recommended, better fill data part)
    :param data: data part of constellations
    :param pilot: pilot part of constellations
    :param data_idx: data index
    :param pilot_idx: pilot index
    :return: constellations mixed pilot and data
    """
    if data.ndim > 1 or pilot.ndim > 1:
        raise ValueError("data and pilot should be 1 dim, details at function add_pilot()")
    assert data.size == data_idx.size, "data.size != data_idx.size"
    assert pilot.size == pilot_idx.size, "data.size != data_idx.size"
    if N is not None:
        n = N//2 - 1
    else:
        n = data.size + pilot.size

    res = np.zeros(n+1)
    if data.size + pilot.size < n:
        padding_idx = np.setdiff1d(np.linspace(1,n,n), np.concatenate([data_idx, pilot_idx]))
        padding = QPSK_mapping(random_bits(padding_idx.size*2).reshape(-1,2), clockwise=padding_clockwise)
        res[padding_idx] = padding
        res[pilot_idx] = pilot
        res[data_idx] = data
    elif data.size + pilot.size == n:
        res[pilot_idx] = pilot
        res[data_idx] = data
    else:
        raise ValueError("given wrong arg, size of data+pilot beyond N")

    return res[1:]


# def image_to_bits(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     print(f"Original image shape: {img.shape}")
#     img = cv2.resize(img, (256,256))  # 减小尺寸
#     _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#     flat = binary_img.flatten()
#     bits = (flat > 0).astype(np.uint8)
#     return bits


def serial_to_parallel(data:np.ndarray, N: int, mode='QPSK'):
    """
        turn serial binary bits into parallel bits,
        make a shape change and pad random bits when a block is not full
    :param data: binary data to be modulated
    :param N: N of OFDM, num of sub carrier waves
    :param mode: constellations mapping mode, supported ['QPSK', ]
    :return: parallel bits, suitable for QPSK mapping or other mapping func
    """
    if mode == 'QPSK':
        q = 2
    else:
        raise ValueError("only support mode ['QPSK',] now")
    n = N//2 -1
    num = np.ceil(data.size / (n*q))
    data = data.flatten()
    if num * n * q > data.size:
        data = np.concatenate([data, random_bits(num*n*q - data.size)])
    data = data.reshape(-1, n, q)
    return data


def random_qpsk_matrix(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """生成 rows×cols 的随机 QPSK（±1±j)/√2 矩阵，用于 OFDM 保护带填充。"""
    re = rng.choice([-1, 1], size=(rows, cols))
    im = rng.choice([-1, 1], size=(rows, cols))
    return (re + 1j * im) / np.sqrt(2)


def prepend_silence_complex(x: np.ndarray, fs: float, silence_sec: float) -> np.ndarray:
    """在复数基带信号最前面加 silence_sec 秒的全零静音。"""
    L = int(round(fs * silence_sec))
    pad = np.zeros(L, dtype=np.complex64 if np.iscomplexobj(x) else np.float64)
    return np.concatenate([pad, x])


def OFDM_modulate_data(
    symbols: np.ndarray,
    N: int,
    cp_len: int,
    *,
    data_start: int = 204,
    data_tail: int = 819,
    fill_seed: int = 2025,
) -> np.ndarray:
    """
    OFDM 调制（保护带 + 共轭对称 + CP）。
    仅在正频的中间 data_bins 承载数据，前后保护带用随机 QPSK 填充。

    参数:
      symbols: 1D QPSK 数据符号（complex）
      N: IFFT 点数
      cp_len: CP 长度
      data_start: 正频前侧保护带宽度（子载波数，默认 204，匹配 receiver 端）
      data_tail: 正频后侧保护带宽度（默认 819）
      fill_seed: 保护带随机 QPSK 的 RNG 种子
    返回:
      with_cp_real: 时域实数波形（1D）
    """
    pos_cnt = N // 2 - 1
    data_bins = pos_cnt - data_start - data_tail
    if data_bins <= 0:
        raise ValueError(f"data_bins <= 0 (pos_cnt={pos_cnt}, data_start={data_start}, data_tail={data_tail})")

    n_sym = len(symbols) // data_bins
    rem = len(symbols) % data_bins
    if rem > 0:
        pad = data_bins - rem
        symbols = np.concatenate([
            symbols,
            QPSK_mapping(np.random.randint(0, 2, size=pad * 2).reshape(-1, 2)),
        ])
        n_sym += 1

    data_matrix = symbols.reshape((n_sym, data_bins))
    freq_data = np.zeros((n_sym, N), dtype=complex)

    data_lo = 1 + data_start
    data_hi = data_lo + data_bins
    freq_data[:, data_lo:data_hi] = data_matrix

    # 保护带随机 QPSK 填充
    rng = np.random.default_rng(fill_seed)
    left_sz = data_lo - 1
    right_sz = (N // 2 - 1) - (data_hi - 1)
    if left_sz > 0:
        freq_data[:, 1 : 1 + left_sz] = random_qpsk_matrix(n_sym, left_sz, rng)
    if right_sz > 0:
        freq_data[:, data_hi : 1 + pos_cnt] = random_qpsk_matrix(n_sym, right_sz, rng)

    # 负频共轭对称
    freq_data[:, N // 2 + 1 :] = np.conj(freq_data[:, 1 : N // 2])[:, ::-1]

    time_data = np.fft.ifft(freq_data, axis=1)
    cp = time_data[:, -cp_len:]
    with_cp = np.hstack([cp, time_data]).flatten()

    max_abs = np.max(np.abs(with_cp))
    if max_abs > 0:
        with_cp = with_cp / max_abs
    return np.real(with_cp)


def ofdm_modulate_symbol(symbol_freq: np.ndarray, cp_len: int = 1024) -> np.ndarray:
    """IFFT + CP for a single OFDM symbol. Returns real time-domain."""
    td = np.fft.ifft(symbol_freq)
    return np.real(np.concatenate([td[-cp_len:], td]))


def OFDM_modulate_data_with_comb(
    symbols: np.ndarray,
    N: int,
    cp_len: int,
    *,
    iteration: int = 5,
    seed: int = 128,
    data_start: int = 204,
    data_tail: int = 819,
    fill_seed: int = 2025,
):
    """
    OFDM modulate + insert comb pilot symbols (no pilot at the very start).
    Guard band sizes match the receiver's data_start / data_tail DATA_BINS.

    Returns:
      with_cp_real: 1D real time-domain waveform
      freq_with_pilot: 2D frequency-domain matrix (data + comb pilots)
    """
    from .batch import generate_pilot_symbol  # lazy import, no circular dep

    pos_cnt = N // 2 - 1
    data_bins = pos_cnt - data_start - data_tail
    if data_bins <= 0:
        raise ValueError(
            f"data_bins <= 0 (pos_cnt={pos_cnt}, data_start={data_start}, data_tail={data_tail})"
        )

    n_sym = len(symbols) // data_bins
    rem = len(symbols) % data_bins
    if rem > 0:
        pad = data_bins - rem
        symbols = np.concatenate([
            symbols,
            QPSK_mapping(np.random.randint(0, 2, size=pad * 2).reshape(-1, 2)),
        ])
        n_sym += 1

    if n_sym == 0:
        return np.array([], dtype=float), np.zeros((0, N), dtype=complex)

    data_matrix = symbols.reshape((n_sym, data_bins))
    freq_data = np.zeros((n_sym, N), dtype=complex)

    data_lo = 1 + data_start
    data_hi = data_lo + data_bins
    freq_data[:, data_lo:data_hi] = data_matrix

    # Guard bands
    rng = np.random.default_rng(fill_seed)
    left_sz = data_lo - 1
    right_sz = (N // 2 - 1) - (data_hi - 1)
    if left_sz > 0:
        freq_data[:, 1 : 1 + left_sz] = random_qpsk_matrix(n_sym, left_sz, rng)
    if right_sz > 0:
        freq_data[:, data_hi : 1 + pos_cnt] = random_qpsk_matrix(n_sym, right_sz, rng)

    # Hermitian symmetry
    freq_data[:, N // 2 + 1 :] = np.conj(freq_data[:, 1 : N // 2])[:, ::-1]

    # Insert comb pilots
    freq_with_pilot_list = []
    pilot_counter = 0
    i = 0
    while i < n_sym:
        for _ in range(iteration):
            if i >= n_sym:
                break
            freq_with_pilot_list.append(freq_data[i])
            i += 1
        if i < n_sym:
            freq_with_pilot_list.append(generate_pilot_symbol(N, seed + pilot_counter))
            pilot_counter += 1

    freq_with_pilot = np.vstack(freq_with_pilot_list)

    # IFFT + CP
    time_data = np.fft.ifft(freq_with_pilot, axis=1)
    cp = time_data[:, -cp_len:]
    with_cp = np.hstack([cp, time_data]).flatten()
    with_cp = np.real(with_cp)
    max_abs = np.max(np.abs(with_cp))
    if max_abs > 0:
        with_cp = with_cp / max_abs
    return with_cp, freq_with_pilot


if __name__ == "__main__":
    # unit test
    test_arr = np.array([[[0,0],[0,1]],
                         [[1,0],[1,1]]])
    print(QPSK_mapping(test_arr))

    data_dir = r'D:\Documents\Coding\Python\SEUCAM\Channel Measurement\data'

    N = 4096
    cp_len = 2000

    bits = get_bits_from_file(os.path.join(data_dir, 'test.txt'))
    bits_parallel = serial_to_parallel(bits, N=N)
    constellations = QPSK_mapping(bits_parallel)
    # save_pilot(constellations,N,data_dir,'test.npy')
    signals = OFDM_modulate(constellations,N=N,cp_len=cp_len)
    # print(bits.shape)
    # print(bits_parallel.shape)
    # print(constellations.shape)
    # print(signals.shape)
    print(random_bits(10))

    contrast()

