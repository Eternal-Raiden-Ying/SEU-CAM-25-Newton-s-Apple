"""
End-to-end codec test: emitter -> direct decode (no channel).
Uses module.emitter.emitter() and module.receiver.receiver_stable.receiver().
Generates waveforms with different parameters, decodes,
verifies output matches original answer.tiff.
"""
import os, sys, io, hashlib, time, argparse
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ / "Channel_Measurement"))
sys.path.insert(0, str(PROJ / "Channel_Measurement" / "module" / "utils" / "ldpc_jossy" / "py"))

from module.emitter import emitter
from module.receiver.receiver_stable import receiver

# ── Paths ──
SAVE_DIR = PROJ / "Channel_Measurement" / "save" / "signal"
OUTPUT_DIR = PROJ / "Channel_Measurement" / "output" / "ldpc"
for d in [SAVE_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

INPUT_FILE = PROJ / "Channel_Measurement" / "data" / "answer.tiff"
assert INPUT_FILE.exists(), f"Input: {INPUT_FILE}"

# ── Fixed params ──
FS, N_FFT, CP_LEN, NUM_PILOT = 48000, 8192, 1024, 8
SUFFIX_MAP = {"tif": "tiff", "txt": "txt", "jpg": "jpg", "png": "png"}


def make_emitter_args(**overrides):
    """Build emitter args with defaults, apply overrides."""
    defaults = dict(
        fs=FS, N=N_FFT, cp_len=CP_LEN, num_pilot=NUM_PILOT,
        chirp_len=2, chirp_l=10, chirp_h=24000, chirp_two=True,
        head_bit=64, size_bit_w=40, type_bit_w=24,
        use_scrambler=False, scrambler_seed=256, scrambler_mode='random',
        ldpc_standard='802.11n', ldpc_rate='1/2', ldpc_z=81, ldpc_ptype='A',
        use_comb=False, comb_iter=10, comb_seed=128,
        data_start=204, data_tail=819, fill_seed=2025,
        pilot_mode='standard',
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_receiver_args(**overrides):
    """Build receiver args with defaults matching receiver.py."""
    defaults = dict(
        fs=FS, N=N_FFT, cp_len=CP_LEN, num_pilot=NUM_PILOT, clockwise=False,
        chirp_len=2, chirp_l=10, chirp_h=24000,
        head_bit=64, size_bit_w=40, type_bit_w=24,
        suffix_map={v: k for k, v in SUFFIX_MAP.items()},
        data_start=204, data_tail=819,
        use_comb=False, INTERVAL=None, COMB_PILOT_SEED_BASE=128,
        edge_expand=32, max_pseudo_iter=20,
        groundtruth=False, tx_file_path=None,
        use_scrambler=False, scrambler_seed=256, scrambler_mode='random',
        scrambler_bitwidth=None,
        ldpc_device='cuda', ldpc_batch=512,
        ldpc_standard='802.11n', ldpc_rate='1/2', ldpc_z=81, ldpc_ptype='A',
        ldpc_microbatch=256, ldpc_llr_clip=10.0, ldpc_max_iter=100,
        ldpc_verbose=False, ldpc_print_iter=False,
        ldpc_log_every=1, ldpc_check_every=1,
        pll_alpha=0.15, pll_beta=0.9, pll_alpha_min=0.05, pll_alpha_max=0.50,
        pll_snr_th_db=6.0, pll_snr_scale=4.0,
        pll_snr_th_min_db=3.0, pll_snr_mid_db=6.0, pll_snr_th_max_db=20.0,
        sig_trk_per_sc=True, sig_trk_alpha_min=0.05, sig_trk_alpha_max=0.7,
        sig_trk_init_sigma=0.3,
        interp_mode='hold', interp_smooth=0.0,
        plot=False,
        plot_opt={'correlation': False, 'impulse_response': False,
                  'raw_pilot_constellation': False, 'corrected_pilot_constellation': False,
                  'data_constellation': False, 'unwrap': False,
                  'received_signal': False, 'BER_show': False,
                  'snr_time_pilot': False, 'snr_time_comb': False,
                  'snr_time_data': False, 'snr_over_sc': False,
                  'freq_offset_interpolate': False},
        print_flag=False,
        print_opt={'pilot_metric': False, 'pilot_delta': False,
                   'data_metric': False, 'iter_verbose': False},
        print_len=64, print_pad='-',
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def fname_from_meta(meta: dict) -> str:
    scram = f"scr{meta['scrambler_seed']}" if meta['scrambler'] else "noscr"
    return (f"tx_N{meta['N']}_cp{meta['cp_len']}_S8{meta['pilot_mode']}"
            f"_R{meta['rate'].replace('/','-')}_Z{meta['Z']}_{scram}.npy")


# ── Test configs ──
TESTS = [
    ("S8standard, no scrambler, Z=81", "standard",  False, 0,  81),
    ("S8same, no scrambler, Z=81",     "same",      False, 0,  81),
    ("S8diff, no scrambler, Z=81",     "different", False, 0,  81),
    ("S8standard, no scrambler, Z=27", "standard",  False, 0,  27),
    ("S8same, random scr=256, Z=81",   "same",      True,  256,81),
]


def main():
    original = INPUT_FILE.read_bytes()
    orig_hash = hashlib.sha256(original).hexdigest()[:16]
    print(f"Input: {INPUT_FILE}  ({len(original)} bytes, sha256={orig_hash})")
    print(f"{'='*60}")

    for desc, pilot_mode, use_scrambler, scr_seed, Z in TESTS:
        print(f"\n-- {desc} --")
        t0 = time.time()

        # TX — use the emitter module
        tx_args = make_emitter_args(
            pilot_mode=pilot_mode, use_scrambler=use_scrambler,
            scrambler_seed=scr_seed, ldpc_z=Z,
        )
        tx_wf, pilots_fd, meta = emitter(str(INPUT_FILE), tx_args)
        tx_time = time.time() - t0

        fname = fname_from_meta(meta)
        np.save(SAVE_DIR / fname, tx_wf)
        print(f"  TX: {len(tx_wf)} samples ({len(tx_wf)/FS:.1f}s) -> {fname}")

        # RX
        t0 = time.time()
        rx_args = make_receiver_args(
            use_scrambler=use_scrambler, scrambler_seed=scr_seed, ldpc_z=Z,
        )
        decoded, info = receiver(tx_wf.astype(np.float64), pilots_fd, rx_args)
        rx_time = time.time() - t0

        # decoded is already payload (receiver strips 64-bit header)
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
