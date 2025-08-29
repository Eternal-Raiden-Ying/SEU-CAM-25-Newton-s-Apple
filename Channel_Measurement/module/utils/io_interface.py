import os
import numpy as np


def get_bits_from_file(file_pth: str) -> np.ndarray:
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
    return bit_array.flatten().astype(np.uint8)


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