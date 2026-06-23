# -*- coding: utf-8 -*-
"""
Emitter with LDPC encoding integrated.
- Reads a TXT/TIFF file as bits
- Scrambles bits
- LDPC-encodes (IEEE 802.11n/802.16 via ldpc_jossy)
- QPSK maps
- OFDM modulates and plays/saves

Dependencies: module/utils/ for shared OFDM/LDPC/I/O functions.
"""
from __future__ import annotations
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from pathlib import Path
import os, sys

# ── Project root & LDPC library ──
_PROJ_ROOT = Path(__file__).resolve().parent
_LDPC_PY_PATH = str(_PROJ_ROOT / "module" / "utils" / "ldpc_jossy" / "py")
if _LDPC_PY_PATH not in sys.path:
    sys.path.append(_LDPC_PY_PATH)
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

try:
    import ldpc  # from ldpc_jossy/py
except Exception as e:
    raise ImportError(
        f"无法导入 ldpc 包：{e}\n"
        f"请检查 LDPC_PY_PATH 是否指向 ldpc_jossy/py 目录。"
    )

# ── Module imports ──
from module.utils.modulate import (
    generate_chirp, QPSK_mapping, OFDM_modulate_data, random_qpsk_matrix,
    prepend_silence_complex, normalize,
)
from module.utils.encode import ldpc_encode_bits, scramble_bits
from module.utils.io_interface import (
    get_bits_from_file, num_to_bits_msb, ascii3_to_24bits,
)
from module.utils.batch import generate_pilot_symbol

# ── Constants ──
fs = 48000
N_fft = 8192
cp_len = 1024
num_pilot_symbols = 8

LDPC_STANDARD = '802.11n'
LDPC_RATE = '1/2'
LDPC_Z = 81
LDPC_PTYPE = 'A'

RANDOM_FILL_SEED = 2025

# ── I/O paths (relative to this file) ──
TXT_INPUT_PATH = str(_PROJ_ROOT / "data" / "shakespace_poem_middle.txt")
TIFF_INPUT_PATH = str(_PROJ_ROOT / "data" / "answer.tiff")
PILOT_SAVE_PATH  = str(_PROJ_ROOT / "save" / "pilot" / "pilot_different_N8192_same.npy")
DATA_WAVEFORM_SAVE_PATH = str(_PROJ_ROOT / "save" / "signal" / "signal_N8192_tiff_nocomb.npy")
WAV_SAVE_PATH = str(_PROJ_ROOT / "save" / "signal" / "tx_signal_N8192_nocomb.wav")


# ═══════════════════════════════════════════════════════════════
#  TX-specific functions (comb pilot, orchestration)
# ═══════════════════════════════════════════════════════════════

def ofdm_modulate_symbol(symbol_freq: np.ndarray, cp_len: int = cp_len) -> np.ndarray:
    """IFFT + CP for one OFDM symbol. Returns real time-domain."""
    time_signal = np.fft.ifft(symbol_freq)
    return np.real(np.concatenate([time_signal[-cp_len:], time_signal]))


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
    OFDM 调制 + 梳状导频插入（开头不插 pilot）。
    保护带大小匹配 receiver 的 data_start / data_tail。

    返回:
      with_cp_real: 时域实数波形（1D）
      freq_with_pilot: 插入 pilot 后的 2D 频域矩阵
    """
    pos_cnt = N // 2 - 1
    data_bins = pos_cnt - data_start - data_tail
    print(f"comb mode: N={N}, data_start={data_start}, data_tail={data_tail}, data_bins={data_bins}")

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
    print(f"总数据符号数: {n_sym} (每符号承载 {data_bins} 子载波)")

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

    # ── Insert comb pilots ──
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
    print(f"插入导频后总符号数: {freq_with_pilot.shape[0]} (原数据符号数: {n_sym})")

    # IFFT + CP
    time_data = np.fft.ifft(freq_with_pilot, axis=1)
    cp = time_data[:, -cp_len:]
    with_cp = np.hstack([cp, time_data]).flatten()
    with_cp = np.real(with_cp)
    max_abs = np.max(np.abs(with_cp))
    if max_abs > 0:
        with_cp = with_cp / max_abs
    return with_cp, freq_with_pilot


def generate_pilot_combed_symbol(
    N: int,
    seed: int = 256,
    iterations: int = 10,
    num_of_data_symbols: int = 50,
    block_size: int = 4,
) -> np.ndarray:
    """Generate concatenated comb pilot sequence (1D)."""
    n_pilots = int(np.ceil(num_of_data_symbols / iterations)) * block_size
    return np.concatenate([generate_pilot_symbol(N, seed + i) for i in range(n_pilots)])


def calculate_papr(signal: np.ndarray) -> tuple[float, float]:
    """Compute PAPR of a time-domain signal. Returns (linear, dB)."""
    power = np.abs(signal) ** 2
    papr = np.max(power) / np.mean(power)
    return papr, 10 * np.log10(papr)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    # Ensure save directories exist
    for p in (PILOT_SAVE_PATH, DATA_WAVEFORM_SAVE_PATH, WAV_SAVE_PATH):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # Chirps
    chirp_sig  = generate_chirp(fs, duration=2, f_l=10, f_h=24000)
    chirp_tail = generate_chirp(fs, duration=2, f_l=20, f_h=18000)
    chirp_cof = 0.4

    tx_signal = np.array([])

    # ── Pilot preamble ──
    mode = input("mode: 1=different, 2=same: ")
    if mode == '1':
        pilot_different = []
        for i in range(num_pilot_symbols):
            p = generate_pilot_symbol(N_fft, seed=256 + i)
            pilot_different.append(p)
            tx_signal = np.concatenate([tx_signal, ofdm_modulate_symbol(p)])
        np.save(PILOT_SAVE_PATH, pilot_different)
    else:
        p = generate_pilot_symbol(N_fft, seed=256)
        pilot_same = []
        for _ in range(num_pilot_symbols):
            pilot_same.append(p)
            tx_signal = np.concatenate([tx_signal, ofdm_modulate_symbol(p)])
        np.save(PILOT_SAVE_PATH, pilot_same)

    tx_signal /= np.max(np.abs(tx_signal))

    # Chirp placement
    ch = input("chirp: 1=two, 2=one: ")
    if ch == '1':
        tx_signal = np.concatenate([chirp_sig * chirp_cof, tx_signal, chirp_tail * chirp_cof])
    else:
        tx_signal = np.concatenate([chirp_sig * chirp_cof, tx_signal])

    # ── Read file ──
    file_path = TIFF_INPUT_PATH
    ext = Path(file_path).suffix.lower()
    bits = get_bits_from_file(file_path)
    print(f"raw bits: {bits.shape}")

    # ── 64-bit header ──
    type_map = {'.txt': 'txt', '.tif': 'tif', '.tiff': 'tif', '.png': 'png'}
    file_type = type_map.get(ext, 'bin')
    hdr_type = ascii3_to_24bits(file_type)
    payload_bit_len = int(len(bits))
    hdr_size = num_to_bits_msb(payload_bit_len, bit_num=40)
    header_64bits = np.concatenate([hdr_type, hdr_size])
    bits = np.concatenate([header_64bits, bits])
    print(f"[TX] header file_type='{file_type}', payload_len(bits)={payload_bit_len}")

    # ── Scrambler ──
    bits_scr = bits.copy().astype(np.uint8)
    # bits_scr = scramble_bits(bits, seed=256, mode='random')  # optional

    # ── LDPC encode ──
    ldpc_bits, (K, Ncw) = ldpc_encode_bits(
        bits_scr, standard=LDPC_STANDARD, rate=LDPC_RATE, z=LDPC_Z, ptype=LDPC_PTYPE,
    )
    print(f"LDPC: K={K}, N={Ncw}, coded length={len(ldpc_bits)}")

    # QPSK
    if len(ldpc_bits) % 2 == 1:
        ldpc_bits = np.concatenate([ldpc_bits, np.array([0], dtype=np.uint8)])
    qpsk_symbols = QPSK_mapping(ldpc_bits.reshape(-1, 2))
    print(f"QPSK symbols: {qpsk_symbols.shape}")

    # ── OFDM modulate ──
    ofdm_mode = input("OFDM mode: 1=comb pilot, 2=no comb: ")
    if ofdm_mode == '1':
        data_waveform, freq_data = OFDM_modulate_data_with_comb(
            qpsk_symbols, N_fft, cp_len, iteration=10, seed=128,
        )
        print(f"freq_data with comb: {freq_data.shape}")
    else:
        data_waveform = OFDM_modulate_data(qpsk_symbols, N_fft, cp_len)

    # ── Assemble final signal ──
    tx_signal_full = np.concatenate([tx_signal, data_waveform])
    tx_signal_full /= np.max(np.abs(tx_signal_full))
    tx_signal_full = np.concatenate([tx_signal_full, chirp_tail * chirp_cof])

    print(f"Final signal: {len(tx_signal_full)} samples, {len(tx_signal_full) / fs:.1f}s")

    # ── Save & play ──
    np.save(DATA_WAVEFORM_SAVE_PATH, tx_signal_full)

    plt.figure(figsize=(12, 6))
    plt.plot(tx_signal_full, label='TX (LDPC-coded)')
    plt.legend(); plt.grid(True); plt.show()

    if WAV_SAVE_PATH:
        write(WAV_SAVE_PATH, fs, (tx_signal_full * 32767).astype(np.int16))

    print("✅ Transmission completed")


if __name__ == "__main__":
    main()
