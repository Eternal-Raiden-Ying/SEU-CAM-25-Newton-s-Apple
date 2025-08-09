import numpy as np
import sounddevice as sd
import os
from scipy.signal import chirp


def generate_chirp(fs, duration, f_l, f_h,*, method='linear'):
    t = np.linspace(0, duration, int(fs * duration))
    chirp_sig = chirp(t, f0=f_l, f1=f_h, t1=duration, method='linear')
    return chirp_sig


def QPSK_mapping(data: np.ndarray,*,clockwise=False):
    """
    turn a binary np.ndarray into a complex sequence, without the conjugate part and cp
    remain its shape except the last dim, e.g. (8,2047,2) -> (8,2047)
    output is not normalized
    :param data: the binary data, the last dim must be 2, either it would raise Error
    :param clockwise:
    :return:
    """
    shape = data.shape
    assert shape[-1] == 2, f"expected length of last dim 2, but received {shape[-1]}"

    dim_order = np.arange(data.ndim).tolist()
    dim_order.pop()
    dim_order.insert(0, data.ndim-1)

    res = np.zeros(shape[:len(shape)-1], dtype=np.complex128)
    bits = np.permute_dims(data, tuple(dim_order))
    b1 = bits[0]
    b0 = bits[1]
    if clockwise:
        res[np.where(b1 == 0)] += 1
        res[np.where(b1 == 1)] -= 1
        res[np.where(b0 == 0)] += 1j
        res[np.where(b0 == 1)] -= 1j
    else:
        res[np.where(b0 == 0)] += 1
        res[np.where(b0 == 1)] -= 1
        res[np.where(b1 == 0)] += 1j
        res[np.where(b1 == 1)] -= 1j
    return res / np.sqrt(2)


def OFDM_modulate(constellations: np.ndarray, N: int, cp_len: int, *, complement_val=0):
    assert constellations.ndim <= 2, "constellations for over 2 dims not supported"
    assert constellations.shape[-1] < N//2, ("in order to have the signal in time domain to be real, "
                                             "length of constellations in each symbol should less than N/2")
    constellation_len = N //2 -1
    num_symbols = int(np.ceil(constellations.size / constellation_len))
    if num_symbols*constellation_len > constellations.size:
        print("Warning! Better make sure constellations.size is k*(N//2-1)")
        complement_len = int(num_symbols*constellation_len - constellations.size)
        constellations = np.concatenate([constellations.flatten(), complement_val*np.ones(complement_len)])
        constellations = constellations.reshape(num_symbols, constellation_len)

    symbols = np.concatenate([np.ones((num_symbols,1), dtype=np.int32), constellations, np.ones((num_symbols,1),dtype=np.int32), np.conjugate(constellations)[:,::-1]],axis=1)

    symbol_td = np.real(np.fft.ifft(symbols, axis=1))
    symbol_with_cp = np.concatenate([symbol_td[:,-cp_len:], symbol_td], axis=1)
    return symbol_with_cp


def normalize(data: np.ndarray):
    if data.ndim == 1:
        max_val = np.max(np.abs(data))
    else:
        max_val = np.max(np.abs(data), axis=-1, keepdims=True)
    return data / max_val

# def image_to_bits(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     print(f"Original image shape: {img.shape}")
#     img = cv2.resize(img, (256,256))  # 减小尺寸
#     _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#     flat = binary_img.flatten()
#     bits = (flat > 0).astype(np.uint8)
#     return bits


def get_bits_from_file(file_pth: str):
    assert os.path.exists(file_pth), f"file not exist, given arg {file_pth}"

    with open(file_pth, 'rb') as file:
        byte_data = file.read()

    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array)
    return bit_array


def serial_to_parallel(data:np.ndarray, N: int, mode='QPSK'):
    """
    pad random bits when a block is not full
    :param data:
    :param N:
    :param mode:
    :return:
    """
    if mode == 'QPSK':
        q = 2
    else:
        raise ValueError("only support mode ['QPSK',] now")
    n = N//2 -1
    num = np.ceil(data.size // N)
    data = data.flatten()
    if num * n * q > data.size:
        data = np.concatenate([data, random_bits(num*n*q - data.size)])
    data = data. reshape(-1, n, q)
    return data


def get_bits_from_str(s: str):
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
    return np.random.randint(low=0, high=2, size=(n,))


def save_pilot(constellations: np.ndarray, N: int, pth, filename):
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
    signals = OFDM_modulate(constellations,N=N,cp_len=cp_len)
    print(bits.shape)
    print(bits_parallel.shape)
    print(constellations.shape)
    print(signals.shape)
