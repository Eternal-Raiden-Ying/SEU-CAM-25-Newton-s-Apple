"""
encode_demo.py — Encode the decoded text back into an OFDM waveform.
Input:  output/ldpc/demo_decoded.txt
Output: assets/signals/demo_reencoded.npy
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

INPUT_PATH  = _PROJ / "output" / "ldpc" / "demo_decoded.txt"
OUTPUT_PATH = _PROJ / "assets" / "signals" / "demo_reencoded.npy"


def main():
    if not INPUT_PATH.exists():
        print(f"Run decode_demo.py first to generate {INPUT_PATH}")
        sys.exit(1)

    cfg = EmitterConfig(pilot_mode="standard")
    cfg.ldpc.device = "cpu"
    cfg.ldpc.max_iter = 5

    t0 = time.time()
    tx_wf, _, _ = emitter(str(INPUT_PATH), cfg)
    elapsed = time.time() - t0

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    np.save(str(OUTPUT_PATH), tx_wf)
    print(f"Encoded: {len(tx_wf)} samples ({len(tx_wf)/cfg.fs:.1f}s, {elapsed:.1f}s)")
    print(f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
