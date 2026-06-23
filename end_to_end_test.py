"""
End-to-end codec test: emitter -> direct decode (no channel).
Uses EmitterConfig for TX, builds Namespace for RX (Phase 2 will migrate RX).
"""
import os, sys, io, hashlib, time
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ / "Channel_Measurement"))
sys.path.insert(0, str(PROJ / "Channel_Measurement" / "module" / "utils" / "ldpc_jossy" / "py"))

from module.emitter import emitter
from module.receiver.receiver_stable import receiver
from module.cfg.config import (
    EmitterConfig, OFDMConfig, ChirpConfig, HeaderConfig,
    ScramblerConfig, LDPCConfig, ReceiverConfig,
)
from module.utils.io_interface import emitter_cfg_to_fname

# ── Paths ──
SIGNALS_DIR = PROJ / "Channel_Measurement" / "assets" / "signals"
OUTPUT_DIR = PROJ / "Channel_Measurement" / "output" / "ldpc"
os.makedirs(SIGNALS_DIR, exist_ok=True)
for d in [SAVE_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

INPUT_FILE = PROJ / "Channel_Measurement" / "data" / "answer.tiff"
SUFFIX_MAP = {"tif": "tiff", "txt": "txt", "jpg": "jpg", "png": "png"}


def emitter_config(pilot_mode: str, use_scrambler: bool, scr_seed: int, Z: int) -> EmitterConfig:
    """Build an EmitterConfig with given overrides."""
    return EmitterConfig(
        ofdm=OFDMConfig(),
        chirp=ChirpConfig(),
        header=HeaderConfig(),
        scrambler=ScramblerConfig(enabled=use_scrambler, seed=scr_seed),
        ldpc=LDPCConfig(z=Z),
        pilot_mode=pilot_mode,
    )


# ── Test configs ──
TESTS = [
    ("S8standard, no scrambler, Z=81", "standard",  False, 0,  81),
    ("S8same, no scrambler, Z=81",     "same",      False, 0,  81),
    ("S8diff, no scrambler, Z=81",     "different", False, 0,  81),
    ("S8standard, no scrambler, Z=27", "standard",  False, 0,  27),
    ("S8same, random scr=256, Z=81",   "same",      True,  256, 81),
]


def main():
    original = INPUT_FILE.read_bytes()
    orig_hash = hashlib.sha256(original).hexdigest()[:16]
    print(f"Input: {INPUT_FILE}  ({len(original)} bytes, sha256={orig_hash})")
    print(f"{'='*60}")

    for desc, pilot_mode, use_scrambler, scr_seed, Z in TESTS:
        print(f"\n-- {desc} --")
        t0 = time.time()

        # TX — EmitterConfig
        tx_cfg = emitter_config(pilot_mode, use_scrambler, scr_seed, Z)
        tx_wf, pilots_fd, meta = emitter(str(INPUT_FILE), tx_cfg)
        tx_time = time.time() - t0

        fname = emitter_cfg_to_fname(tx_cfg)
        np.save(SIGNALS_DIR / fname, tx_wf)
        print(f"  TX: {len(tx_wf)} samples ({len(tx_wf)/48000:.1f}s) -> {fname}")

        # RX — ReceiverConfig directly
        t0 = time.time()
        rx_cfg = ReceiverConfig(
            scrambler=ScramblerConfig(enabled=use_scrambler, seed=scr_seed),
            ldpc=LDPCConfig(z=Z),
        )
        decoded, info = receiver(tx_wf.astype(np.float64), pilots_fd, rx_cfg)
        rx_time = time.time() - t0

        out_bytes = np.packbits(decoded.flatten()).tobytes()
        match = (out_bytes == original)

        suffix = meta['file_type']
        out_path = OUTPUT_DIR / f"e2e_{fname.replace('.npy', '')}.{suffix}"
        with open(out_path, 'wb') as f:
            f.write(out_bytes)

        status = "OK" if match else "MISMATCH"
        print(f"  RX: iter={info['ldpc_iter']}, payload={len(out_bytes)}B "
              f"(orig={len(original)}B), match={match}, type={info.get('type_suffix','?')}")
        print(f"  [{status}] tx={tx_time:.1f}s rx={rx_time:.1f}s -> {out_path.name}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
