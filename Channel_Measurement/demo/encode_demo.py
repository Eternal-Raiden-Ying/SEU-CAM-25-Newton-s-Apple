"""
encode_demo.py — Encode a text file into an OFDM waveform (.npy).
Usage: python encode_demo.py [path_to_txt]

Output: assets/signals/tx_fs48000_*.npy
"""
import os, sys, io, time
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "module" / "utils" / "ldpc_jossy" / "py"))

from module.emitter import emitter
from module.cfg.config import EmitterConfig
from module.utils.io_interface import emitter_cfg_to_fname

SIGNALS_DIR = _PROJ / "assets" / "signals"
DEMO_TXT = _PROJ / "data" / "demo.txt"


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else str(DEMO_TXT)
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        print("Usage: python encode_demo.py <path_to_txt>")
        sys.exit(1)
    input_path = str(Path(input_path).resolve())

    original = Path(input_path).read_bytes()
    print(f"Input: {input_path}  ({len(original)} bytes)")

    cfg = EmitterConfig(pilot_mode="standard")
    cfg.ldpc.device = "cpu"
    cfg.ldpc.max_iter = 5

    t0 = time.time()
    tx_wf, pilots_fd, meta = emitter(input_path, cfg)
    tx_time = time.time() - t0

    os.makedirs(SIGNALS_DIR, exist_ok=True)
    fname = emitter_cfg_to_fname(cfg)
    np.save(SIGNALS_DIR / fname, tx_wf)

    print(f"Encoded: {len(tx_wf)} samples ({len(tx_wf)/cfg.fs:.1f}s, {tx_time:.1f}s)")
    print(f"Output: assets/signals/{fname}")


if __name__ == "__main__":
    main()
