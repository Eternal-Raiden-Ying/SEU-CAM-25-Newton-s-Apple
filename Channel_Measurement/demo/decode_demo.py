"""
decode_demo.py — Decode an OFDM waveform (.npy) back to a text file.
Usage: python decode_demo.py [path_to_npy]

Output: output/ldpc/demo_*.txt
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
from module.utils.io_interface import fname_to_emitter_cfg, emitter_cfg_to_fname

SIGNALS_DIR = _PROJ / "assets" / "signals"
PILOTS_DIR = _PROJ / "assets" / "pilots"
OUTPUT_DIR = _PROJ / "output" / "ldpc"


def find_latest_signal():
    """Find the most recent .npy signal file in assets/signals/."""
    files = sorted(SIGNALS_DIR.glob("*.npy"), key=os.path.getmtime, reverse=True)
    return str(files[0]) if files else None


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_signal()
    if input_path is None or not os.path.exists(input_path):
        print("No signal file found. Run encode_demo.py first, or specify a path.")
        print("Usage: python decode_demo.py <path_to_npy>")
        sys.exit(1)
    input_path = str(Path(input_path).resolve())
    print(f"Loading: {input_path}")

    rx = np.load(input_path).ravel().astype(np.float64)

    # Get matching pilot
    try:
        tx_cfg = fname_to_emitter_cfg(os.path.basename(input_path))
    except Exception:
        tx_cfg = None
    pilot_name = f"pilot_STANDARD_freq_domain.npy"
    pilot = np.load(PILOTS_DIR / pilot_name)

    cfg = ReceiverConfig()
    cfg.ldpc.device = "cpu"
    cfg.ldpc.max_iter = 5

    t0 = time.time()
    decoded, info = receiver(rx, pilot, cfg)
    rx_time = time.time() - t0

    out_bytes = np.packbits(decoded.flatten()).tobytes()
    suffix = info.get("type_suffix", "txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = OUTPUT_DIR / f"demo_decoded.{suffix}"
    with open(out_path, "wb") as f:
        f.write(out_bytes)

    print(f"Decoded: {len(out_bytes)} bytes, iter={info['ldpc_iter']}, "
          f"type={suffix}, time={rx_time:.1f}s")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
