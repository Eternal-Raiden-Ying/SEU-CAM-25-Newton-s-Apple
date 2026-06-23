"""
Demo: encode a text file → decode → verify.
Usage: python demo.py [path_to_txt_file]

If no file is given, creates a sample txt and runs on that.
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
from module.cfg.config import EmitterConfig, ReceiverConfig, ScramblerConfig, LDPCConfig
from module.utils.io_interface import emitter_cfg_to_fname

# ── Paths ──
ASSETS_DIR = PROJ / "Channel_Measurement" / "assets"
SIGNALS_DIR = ASSETS_DIR / "signals"
OUTPUT_DIR = PROJ / "Channel_Measurement" / "output" / "ldpc"
for d in [SIGNALS_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

DEMO_TXT = PROJ / "demo.txt"


def main():
    # Determine input file
    input_path = sys.argv[1] if len(sys.argv) > 1 else str(DEMO_TXT)
    if not Path(input_path).exists():
        print(f"File not found: {input_path}")
        print(f"Create a demo.txt or run: python demo.py <file>")
        sys.exit(1)
    input_path = str(Path(input_path).resolve())
    original = Path(input_path).read_bytes()
    orig_hash = hashlib.sha256(original).hexdigest()[:16]
    print(f"Input: {input_path}  ({len(original)} bytes, sha256={orig_hash})")

    # ── Encode ──
    tx_cfg = EmitterConfig(pilot_mode="standard")
    print(f"Encoding with: N={tx_cfg.N}, cp={tx_cfg.cp_len}, "
          f"pilot={tx_cfg.pilot_mode}, Z={tx_cfg.ldpc_z}")

    t0 = time.time()
    tx_wf, pilots_fd, meta = emitter(input_path, tx_cfg)
    tx_time = time.time() - t0

    fname = emitter_cfg_to_fname(tx_cfg)
    np.save(SIGNALS_DIR / fname, tx_wf)
    print(f"  TX: {len(tx_wf)} samples ({len(tx_wf)/48000:.1f}s) -> {fname}")

    # ── Decode ──
    rx_cfg = ReceiverConfig()
    t0 = time.time()
    decoded, info = receiver(tx_wf.astype(np.float64), pilots_fd, rx_cfg)
    rx_time = time.time() - t0

    out_bytes = np.packbits(decoded.flatten()).tobytes()
    match = (out_bytes == original)

    suffix = meta["file_type"]
    out_path = OUTPUT_DIR / f"demo_{fname.replace('.npy', '')}.{suffix}"
    with open(out_path, "wb") as f:
        f.write(out_bytes)

    print(f"  RX: iter={info['ldpc_iter']}, payload={len(out_bytes)}B, "
          f"match={match}, type={info.get('type_suffix', '?')}")
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] tx={tx_time:.1f}s rx={rx_time:.1f}s -> {out_path.name}")


if __name__ == "__main__":
    main()
