import numpy as np
import sounddevice as sd
import os
from scipy.signal import chirp
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
                  complement_val=0, padding_clockwise=False):
    """
        given constellations and modulate into time signal (with cyclic prefix)
        if use comb-type pilot or block-type pilot, turn to add_pilot() first

        v2: complement_val deprecated, extreme high peak due to continuous identical values,
            I just noticed that, maybe not that reason exactly, but I think constellations
            from random bits is more reasonable
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
        padding_constellations = QPSK_mapping(random_bits(2*complement_len).reshape(-1,2),clockwise=padding_clockwise)
        constellations = np.concatenate([constellations.flatten(), padding_constellations])
        constellations = constellations.reshape(num_symbols, constellation_len)

    symbols = np.concatenate([np.ones((num_symbols,1), dtype=np.int32), constellations, np.ones((num_symbols,1),dtype=np.int32), np.conjugate(constellations)[:,::-1]],axis=1)

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


def get_bits_from_file(file_pth: str):
    """
        read file in binary and return a binary np.ndarray (flattened)
    :param file_pth: just file path
    :return: binary np.ndarray, like [0,0,0,1,1,1,.....]
    """
    assert os.path.exists(file_pth), f"file not exist, given arg {file_pth}"

    with open(file_pth, 'rb') as file:
        byte_data = file.read()

    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array)
    return bit_array.flatten()


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
    num = np.ceil(data.size / N)
    data = data.flatten()
    if num * n * q > data.size:
        data = np.concatenate([data, random_bits(num*n*q - data.size)])
    data = data. reshape(-1, n, q)
    return data


def get_bits_from_str(s: str):
    """
        get bits from given string
    :param s: string
    :return: bits, in data type np.ndarray (binary)
    """
    byte_data = s.encode('utf-8')
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array)
    return bit_array


def random_bits(n: int):
    """
        return a random generated binary ndarray with size n
    :param n:
    :return:
    """
    return np.random.randint(low=0, high=2, size=(int(n),))


def save_pilot(constellations: np.ndarray, N: int, pth, filename):
    """
        save pilot / freq data, for sender
    :param constellations: constellations (data + pilot)
    :param N: num of sub carrier waves
    :param pth: directory for file to save (create automatically if not exists)
    :param filename: filename
    :return:
    """
    constellation_len = N // 2 - 1
    num_symbols = int(np.ceil(constellations.size / constellation_len))
    if num_symbols * constellation_len > constellations.size:
        complement_len = int(num_symbols * constellation_len - constellations.size)
        constellations = np.concatenate([constellations.flatten(), 0 * np.ones(complement_len)])
        constellations = constellations.reshape(num_symbols, constellation_len)

    pilots = np.concatenate(
        [np.ones((num_symbols, 1), dtype=np.int32), constellations, np.ones((num_symbols, 1), dtype=np.int32),
         np.conjugate(constellations)[:, ::-1]], axis=1)
    if not os.path.exists(pth):
        os.makedirs(pth)
    np.save(os.path.join(pth, filename), pilots)


def contrast():
    # 录音播放部分
    import numpy as np
    import sounddevice as sd
    from scipy.io.wavfile import write
    from scipy.signal import chirp
    import matplotlib.pyplot as plt
    import numpy as np
    import sounddevice as sd
    from scipy.io.wavfile import write
    from scipy.signal import chirp
    import matplotlib.pyplot as plt
    import os

    # Generate a linear chirp signal
    def generate_chirp(fs, duration=2, f0=10, f1=24000):
        t = np.linspace(0, duration, int(fs * duration))
        chirp_sig = chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
        return chirp_sig

    # Parameters
    fs = 48000
    N = 4096
    cp_len = 2000
    num_symbols = 16

    # Generate pilot symbol (e.g., using BPSK or QPSK)
    def generate_pilot_symbol(N):
        half = N // 2

        real_parts = np.random.choice([-1, 1], size=half - 1)
        imag_parts = np.random.choice([-1, 1], size=half - 1)
        X_half = (real_parts + 1j * imag_parts) / np.sqrt(2)

        X_freq = np.zeros(N, dtype=complex)
        X_freq[0] = 1                                                                                # DC component
        X_freq[1:half] = X_half
        X_freq[half] = 1                                                                    # Nyquist frequency (real)
        X_freq[half + 1:] = np.conj(X_half[::-1])        # Hermitian symmetry
        return X_freq

    # Perform OFDM modulation with IFFT and cyclic prefix
    def ofdm_modulate(symbol_freq):
        time_signal = np.fft.ifft(symbol_freq)
        return np.concatenate([time_signal[-cp_len:], time_signal])

    # Generate the chirp signal for prefix
    chirp_sig = generate_chirp(fs)
    chirp_tail = generate_chirp(fs, f0=20, f1=24000)

    tx_signal = np.array([])

    pilots = []
    for _ in range(num_symbols):
        pilot = generate_pilot_symbol(N)
        pilots.append(pilot[1:N//2])
        ofdm_time = ofdm_modulate(pilot)
        tx_signal = np.concatenate([tx_signal, ofdm_time])

    my_tx = OFDM_modulate(constellations=np.array(pilots),N=N, cp_len=cp_len)
                   

    # Convert to real signal and normalize
    tx_signal_real = np.real(tx_signal)

    tx_signal_real /= np.max(np.abs(tx_signal_real))
    my_tx = normalize(my_tx.flatten())
    tx_signal_real = np.concatenate([chirp_sig, tx_signal_real])
    return tx_signal_real


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
