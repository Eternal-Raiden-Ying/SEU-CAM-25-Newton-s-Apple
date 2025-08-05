import numpy as np
import sounddevice as sd
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
    :param data:
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
    return res


def OFDM_modulate(constellations: np.ndarray, N: int, cp_len: int):
    assert constellations.ndim <= 2, "constellations for over 2 dims not supported"
    assert constellations.shape[-1] < N//2, ("in order to have the signal in time domain to be real, "
                                             "length of constellations in each symbol should less than N/2")
    constellation_len = N //2 -1
    num_symbols = np.ceil(constellations.size / constellation_len)
    constellations = np.concatenate([constellations.flatten(), np.zeros(num_symbols*constellation_len - constellations.size)])
    constellations = constellations.reshape(num_symbols, constellation_len)

    symbols = np.concatenate([np.ones((num_symbols,1)), constellations, np.ones((num_symbols,1)), np.conjugate(constellations)[::-1]],axis=0)

    symbol_td = np.fft.ifft(symbols, axis=1)
    symbol_with_cp = np.concatenate(symbol_td, symbol_td[:,-cp_len:], axis=1)
    return symbol_with_cp


def normalize(data: np.ndarray):
    if data.ndim == 1:
        max_val = np.max(np.abs(data))
    else:
        max_val = np.max(np.abs(data), axis=-1)
    return data / max_val

# def image_to_bits(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     print(f"Original image shape: {img.shape}")
#     img = cv2.resize(img, (256,256))  # 减小尺寸
#     _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#     flat = binary_img.flatten()
#     bits = (flat > 0).astype(np.uint8)
#     return bits



if __name__ == "__main__":
    # unit test
    test_arr = np.array([[[0,0],[0,1]],
                         [[1,0],[1,1]]])
    print(QPSK_mapping(test_arr))
