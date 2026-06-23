# -*- coding: utf-8 -*-
"""
Emitter CLI — thin wrapper around module.emitter.emitter().
Configures an EmitterConfig dataclass, calls emitter(), saves/plays the waveform.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from pathlib import Path
import os, sys

_PROJ_ROOT = Path(__file__).resolve().parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from module.emitter import emitter
from module.cfg.config import EmitterConfig, OFDMConfig, ChirpConfig, HeaderConfig, ScramblerConfig, LDPCConfig

# ── Default paths ──
TIFF_INPUT_PATH = str(_PROJ_ROOT / "data" / "answer.tiff")
TXT_INPUT_PATH  = str(_PROJ_ROOT / "data" / "shakespace_poem_middle.txt")
PILOT_SAVE_PATH = str(_PROJ_ROOT / "assets" / "pilots" / "pilot_different_N8192_same.npy")
DATA_WAVEFORM_SAVE_PATH = str(_PROJ_ROOT / "assets" / "signals" / "signal_N8192_tiff_nocomb.npy")
WAV_SAVE_PATH = str(_PROJ_ROOT / "assets" / "signals" / "tx_signal_N8192_nocomb.wav")


def main():
    # Interactive prompts
    mode = input("pilot mode: 1=different, 2=same: ")
    pilot_mode = 'different' if mode == '1' else 'same'

    ch = input("chirp: 1=two, 2=one: ")
    two_chirp = (ch == '1')

    ofdm_mode = input("OFDM mode: 1=comb pilot, 2=no comb: ")
    use_comb = (ofdm_mode == '1')

    # Build config with defaults (all match production values)
    cfg = EmitterConfig(
        ofdm=OFDMConfig(),
        chirp=ChirpConfig(two_chirp=two_chirp),
        header=HeaderConfig(),
        scrambler=ScramblerConfig(),
        ldpc=LDPCConfig(),
        pilot_mode=pilot_mode,
        comb_enabled=use_comb,
    )

    # Ensure save dirs
    for p in (PILOT_SAVE_PATH, DATA_WAVEFORM_SAVE_PATH, WAV_SAVE_PATH):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # ── Build waveform ──
    file_path = TIFF_INPUT_PATH
    print(f"Encoding: {file_path}")
    tx, pilots_fd, meta = emitter(file_path, cfg)

    print(f"TX: {meta['tx_len']} samples, {meta['tx_len']/cfg.fs:.1f}s")
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
        write(WAV_SAVE_PATH, cfg.fs, (tx * 32767).astype(np.int16))

    print("Done.")


if __name__ == "__main__":
    main()
