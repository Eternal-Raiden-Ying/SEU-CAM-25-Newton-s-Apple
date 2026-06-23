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


def num_to_bits_msb(n: int, *, bit_num: int = 40) -> np.ndarray:
    """
    Convert integer n to bit_num bits (MSB-first). Used for file size in bits.
    """
    assert 0 <= n < (1 << bit_num), f"payload length must fit in {bit_num} bits"
    b = np.zeros(bit_num, dtype=np.uint8)
    for k in range(bit_num):
        b[k] = (n >> (bit_num - 1 - k)) & 1
    return b


def ascii3_to_24bits(s3: str) -> np.ndarray:
    """
    Encode 3 ASCII chars to 24 bits (MSB-first).
    e.g. 'txt' / 'tif' / 'bin' -- for the 64-bit file header type field.
    """
    assert len(s3) == 3, "file type must be 3 chars"
    val = (ord(s3[0]) << 16) | (ord(s3[1]) << 8) | ord(s3[2])
    b = np.zeros(24, dtype=np.uint8)
    for k in range(24):
        b[k] = (val >> (23 - k)) & 1
    return b
