"""
decode_demo.py — Decode the demo signal back to a text file.
Output: output/ldpc/demo_decoded.txt
"""
import os, sys, io, time
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "module" / "utils" / "ldpc_jossy" / "py"))

from module.receiver.receiver_stable import receiver
from module.cfg.config import ReceiverConfig

SIGNAL_PATH = _PROJ / "assets" / "signals" / "demo_signal.npy"
PILOT_PATH  = _PROJ / "assets" / "pilots" / "pilot_STANDARD_freq_domain.npy"
OUTPUT_DIR  = _PROJ / "output" / "ldpc"


def main():
    rx = np.load(str(SIGNAL_PATH)).ravel().astype(np.float64)
    pilot = np.load(str(PILOT_PATH))

    cfg = ReceiverConfig()
    cfg.ldpc.device = "cpu"
    cfg.ldpc.max_iter = 5

    t0 = time.time()
    decoded, info = receiver(rx, pilot, cfg)
    elapsed = time.time() - t0

    out_bytes = np.packbits(decoded.flatten()).tobytes()
    suffix = info.get("type_suffix", "txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = OUTPUT_DIR / f"demo_decoded.{suffix}"
    with open(out_path, "wb") as f:
        f.write(out_bytes)

    print(f"Decoded: {len(out_bytes)}B, iter={info['ldpc_iter']}, {elapsed:.1f}s")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
