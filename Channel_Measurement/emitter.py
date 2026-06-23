# -*- coding: utf-8 -*-
"""
Emitter CLI — thin wrapper around module.emitter.emitter().
Configures parameters, calls emitter(), saves/plays the waveform.
"""
from __future__ import annotations
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from pathlib import Path
import os, sys

# Project root & module path
_PROJ_ROOT = Path(__file__).resolve().parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from module.emitter import emitter

# ── Default paths ──
TIFF_INPUT_PATH = str(_PROJ_ROOT / "data" / "answer.tiff")
TXT_INPUT_PATH  = str(_PROJ_ROOT / "data" / "shakespace_poem_middle.txt")
PILOT_SAVE_PATH = str(_PROJ_ROOT / "save" / "pilot" / "pilot_different_N8192_same.npy")
DATA_WAVEFORM_SAVE_PATH = str(_PROJ_ROOT / "save" / "signal" / "signal_N8192_tiff_nocomb.npy")
WAV_SAVE_PATH = str(_PROJ_ROOT / "save" / "signal" / "tx_signal_N8192_nocomb.wav")


def make_emitter_args() -> argparse.Namespace:
    """Default emitter arguments (matching receiver.py defaults)."""
    suffix_map = {"tif": "tiff", "txt": "txt", "jpg": "jpg", "png": "png"}
    return argparse.Namespace(
        fs=48000, N=8192, cp_len=1024, num_pilot=8,
        chirp_len=2, chirp_l=10, chirp_h=24000, chirp_two=True,
        head_bit=64, size_bit_w=40, type_bit_w=24,
        suffix_map={v: k for k, v in suffix_map.items()},
        use_scrambler=False, scrambler_seed=256, scrambler_mode='random',
        ldpc_standard='802.11n', ldpc_rate='1/2', ldpc_z=81, ldpc_ptype='A',
        use_comb=False, comb_iter=10, comb_seed=128,
        data_start=204, data_tail=819, fill_seed=2025,
        pilot_mode='different',
    )


def main():
    args = make_emitter_args()

    # Interactive prompts
    mode = input("pilot mode: 1=different, 2=same: ")
    args.pilot_mode = 'different' if mode == '1' else 'same'

    ch = input("chirp: 1=two, 2=one: ")
    args.chirp_two = (ch == '1')

    ofdm_mode = input("OFDM mode: 1=comb pilot, 2=no comb: ")
    args.use_comb = (ofdm_mode == '1')

    # Ensure save dirs
    for p in (PILOT_SAVE_PATH, DATA_WAVEFORM_SAVE_PATH, WAV_SAVE_PATH):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # ── Build waveform ──
    file_path = TIFF_INPUT_PATH
    print(f"Encoding: {file_path}")
    tx, pilots_fd, meta = emitter(file_path, args)

    print(f"TX: {meta['tx_len']} samples, {meta['tx_len']/args.fs:.1f}s")
    print(f"LDPC: K={meta['K']}, Ncw={meta['Ncw']}, Z={meta['Z']}")
    print(f"Pilot: {meta['pilot_mode']}, Scrambler: {meta['scrambler']}, Comb: {meta['comb']}")

    # Save
    np.save(DATA_WAVEFORM_SAVE_PATH, tx)
    np.save(PILOT_SAVE_PATH, pilots_fd)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(tx, label='TX (LDPC-coded)')
    plt.legend(); plt.grid(True); plt.show()

    if WAV_SAVE_PATH:
        write(WAV_SAVE_PATH, args.fs, (tx * 32767).astype(np.int16))

    print("Done.")


if __name__ == "__main__":
    main()
