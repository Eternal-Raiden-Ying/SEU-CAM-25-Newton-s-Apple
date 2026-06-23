"""
End-to-end codec test: emitter -> direct decode (no channel).
Uses EmitterConfig for TX, builds Namespace for RX (Phase 2 will migrate RX).
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
from module.cfg.config import (
    EmitterConfig, OFDMConfig, ChirpConfig, HeaderConfig,
    ScramblerConfig, LDPCConfig, ReceiverConfig,
)

# ── Paths ──
SAVE_DIR = PROJ / "Channel_Measurement" / "save" / "signal"
OUTPUT_DIR = PROJ / "Channel_Measurement" / "output" / "ldpc"
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


def receiver_namespace(cfg: ReceiverConfig) -> argparse.Namespace:
    """Build an argparse.Namespace from a ReceiverConfig (Phase 2 will remove this bridge)."""
    return argparse.Namespace(
        fs=cfg.fs, N=cfg.N, cp_len=cfg.cp_len, num_pilot=cfg.num_pilot,
        clockwise=cfg.clockwise,
        chirp_len=cfg.chirp_len, chirp_l=cfg.chirp_l, chirp_h=cfg.chirp_h,
        head_bit=cfg.head_bit, size_bit_w=cfg.size_bit_w, type_bit_w=cfg.type_bit_w,
        suffix_map=cfg.suffix_map_rev,
        data_start=cfg.data_start, data_tail=cfg.data_tail,
        use_comb=cfg.use_comb, INTERVAL=cfg.INTERVAL, COMB_PILOT_SEED_BASE=cfg.COMB_PILOT_SEED_BASE,
        edge_expand=cfg.edge_expand, max_pseudo_iter=cfg.max_pseudo_iter,
        groundtruth=cfg.groundtruth, tx_file_path=cfg.tx_file_path,
        use_scrambler=cfg.use_scrambler, scrambler_seed=cfg.scrambler_seed,
        scrambler_mode=cfg.scrambler_mode, scrambler_bitwidth=cfg.scrambler_bitwidth,
        ldpc_device=cfg.ldpc_device, ldpc_batch=cfg.ldpc_batch,
        ldpc_standard=cfg.ldpc_standard, ldpc_rate=cfg.ldpc_rate,
        ldpc_z=cfg.ldpc_z, ldpc_ptype=cfg.ldpc_ptype, ldpc_microbatch=cfg.ldpc_microbatch,
        ldpc_llr_clip=cfg.ldpc_llr_clip, ldpc_max_iter=cfg.ldpc_max_iter,
        ldpc_verbose=cfg.ldpc_verbose, ldpc_print_iter=cfg.ldpc_print_iter,
        ldpc_log_every=cfg.ldpc_log_every, ldpc_check_every=cfg.ldpc_check_every,
        pll_alpha=cfg.pll_alpha, pll_beta=cfg.pll_beta,
        pll_alpha_min=cfg.pll_alpha_min, pll_alpha_max=cfg.pll_alpha_max,
        pll_snr_th_db=cfg.pll_snr_th_db, pll_snr_scale=cfg.pll_snr_scale,
        pll_snr_th_min_db=cfg.pll_snr_th_min_db, pll_snr_mid_db=cfg.pll_snr_mid_db,
        pll_snr_th_max_db=cfg.pll_snr_th_max_db,
        sig_trk_per_sc=cfg.sig_trk_per_sc, sig_trk_alpha_min=cfg.sig_trk_alpha_min,
        sig_trk_alpha_max=cfg.sig_trk_alpha_max, sig_trk_init_sigma=cfg.sig_trk_init_sigma,
        interp_mode=cfg.interp_mode, interp_smooth=cfg.interp_smooth,
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

        fname = fname_from_meta(meta)
        np.save(SAVE_DIR / fname, tx_wf)
        print(f"  TX: {len(tx_wf)} samples ({len(tx_wf)/48000:.1f}s) -> {fname}")

        # RX — ReceiverConfig → Namespace bridge (Phase 2 will remove)
        t0 = time.time()
        rx_cfg = ReceiverConfig(
            scrambler=ScramblerConfig(enabled=use_scrambler, seed=scr_seed),
            ldpc=LDPCConfig(z=Z),
        )
        rx_args = receiver_namespace(rx_cfg)
        decoded, info = receiver(tx_wf.astype(np.float64), pilots_fd, rx_args)
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
