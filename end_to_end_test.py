"""
End-to-end codec test: emitter -> direct decode (no channel).
Generates waveforms with different parameters, decodes,
verifies output matches original answer.tiff.
Saves waveforms with parameter-encoded filenames.
"""
import os, sys, io, hashlib, time, argparse
import numpy as np
from pathlib import Path
from scipy.signal import chirp

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ / "Channel_Measurement"))
sys.path.insert(0, str(PROJ / "Channel_Measurement" / "module" / "utils" / "ldpc_jossy" / "py"))

from module.receiver.receiver_stable import receiver
from module.utils.encode import ldpc_encode_bits, ldpc_make_code, scramble_bits
from module.utils.modulate import QPSK_mapping, generate_chirp
from module.utils.io_interface import get_bits_from_file, num_to_bits_msb

# ── Paths ──
DATA_DIR = PROJ / "Channel_Measurement" / "data"
SAVE_DIR = PROJ / "Channel_Measurement" / "save" / "signal"
OUTPUT_DIR = PROJ / "Channel_Measurement" / "output" / "ldpc"
for d in [SAVE_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

INPUT_FILE = DATA_DIR / "answer.tiff"
assert INPUT_FILE.exists(), f"Input: {INPUT_FILE}"

# ── Fixed params ──
FS, N_FFT, CP_LEN, NUM_PILOT = 48000, 8192, 1024, 8
SYM_LEN = N_FFT + CP_LEN
DATA_START, DATA_TAIL = 204, 819
POS_CNT = N_FFT // 2 - 1
DATA_BINS_CNT = POS_CNT - DATA_START - DATA_TAIL
SUFFIX_MAP = {"tif": "tiff", "txt": "txt", "jpg": "jpg", "png": "png"}


def make_args(**overrides):
    """Build receiver args Namespace with defaults matching receiver.py."""
    defaults = dict(
        fs=FS, N=N_FFT, cp_len=CP_LEN, num_pilot=NUM_PILOT, clockwise=False,
        chirp_len=2, chirp_l=10, chirp_h=24000,
        head_bit=64, size_bit_w=40, type_bit_w=24,
        suffix_map={v: k for k, v in SUFFIX_MAP.items()},
        data_start=DATA_START, data_tail=DATA_TAIL,
        use_comb=False, INTERVAL=None, COMB_PILOT_SEED_BASE=128,
        edge_expand=32, max_pseudo_iter=20, first_try_portion=1.0,
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


def gen_pilot_fd(N, seed):
    """Generate one OFDM pilot symbol (frequency domain)."""
    rng = np.random.default_rng(seed)
    half = N // 2
    re = rng.choice([-1, 1], size=half - 1)
    im = rng.choice([-1, 1], size=half - 1)
    X_half = (re + 1j * im) / np.sqrt(2)
    X = np.zeros(N, dtype=complex)
    X[1:half] = X_half
    X[half + 1:] = np.conj(X_half[::-1])
    return X


def ofdm_modulate_pilot(pilot_fd):
    """IFFT + CP for one pilot symbol. Returns real time-domain."""
    td = np.fft.ifft(pilot_fd)
    return np.real(np.concatenate([td[-CP_LEN:], td]))


def ofdm_modulate_data(qpsk_symbols, fill_seed=2025):
    """
    OFDM modulate QPSK data symbols.
    Places data in bins matching receiver's DATA_START/DATA_TAIL.
    Returns real time-domain waveform.
    """
    data_bins = DATA_BINS_CNT
    n_sym = len(qpsk_symbols) // data_bins
    rem = len(qpsk_symbols) % data_bins
    if rem > 0:
        pad = data_bins - rem
        qpsk_symbols = np.concatenate([
            qpsk_symbols,
            QPSK_mapping(np.random.randint(0, 2, size=pad * 2).reshape(-1, 2))
        ])
        n_sym += 1

    dm = qpsk_symbols.reshape(n_sym, data_bins)
    fd = np.zeros((n_sym, N_FFT), dtype=complex)

    data_lo = 1 + DATA_START
    data_hi = data_lo + data_bins
    fd[:, data_lo:data_hi] = dm

    # Random QPSK fill for guard bands
    rng = np.random.default_rng(fill_seed)
    left_sz = data_lo - 1
    right_sz = (N_FFT // 2 - 1) - (data_hi - 1)
    if left_sz > 0:
        re = rng.choice([-1, 1], size=(n_sym, left_sz))
        im = rng.choice([-1, 1], size=(n_sym, left_sz))
        fd[:, 1:1 + left_sz] = (re + 1j * im) / np.sqrt(2)
    if right_sz > 0:
        re = rng.choice([-1, 1], size=(n_sym, right_sz))
        im = rng.choice([-1, 1], size=(n_sym, right_sz))
        fd[:, data_hi:1 + POS_CNT] = (re + 1j * im) / np.sqrt(2)

    # Hermitian symmetry
    fd[:, N_FFT // 2 + 1:] = np.conj(fd[:, 1:N_FFT // 2])[:, ::-1]

    time_data = np.fft.ifft(fd, axis=1)
    cp = time_data[:, -CP_LEN:]
    with_cp = np.hstack([cp, time_data]).flatten()
    max_abs = np.max(np.abs(with_cp))
    if max_abs > 0:
        with_cp = with_cp / max_abs
    return np.real(with_cp)


def build_tx(file_path: Path, pilot_mode: str, use_scrambler: bool,
             scrambler_seed: int, Z: int, rate: str = "1/2"):
    """
    Build complete transmit waveform.
    Returns: (tx_waveform, pilots_fd, meta_dict)
    """
    # ── Chirps ──
    chirp_front = generate_chirp(FS, duration=2, f_l=10, f_h=24000) * 0.4
    chirp_tail = generate_chirp(FS, duration=2, f_l=20, f_h=18000) * 0.4

    # ── Pilot preamble ──
    if pilot_mode == "standard":
        std_path = PROJ / "Channel_Measurement" / "save" / "pilot" / "pilot_STANDARD_freq_domain.npy"
        if std_path.exists():
            pilots_fd = np.load(std_path)
        else:
            pilots_fd = np.array([gen_pilot_fd(N_FFT, 256 + i) for i in range(NUM_PILOT)])
    elif pilot_mode == "different":
        pilots_fd = np.array([gen_pilot_fd(N_FFT, 256 + i) for i in range(NUM_PILOT)])
    else:  # "same"
        p = gen_pilot_fd(N_FFT, 256)
        pilots_fd = np.array([p] * NUM_PILOT)

    tx_preamble = np.concatenate([ofdm_modulate_pilot(p) for p in pilots_fd])
    tx_preamble /= np.max(np.abs(tx_preamble))

    # ── Read file + header ──
    bits = get_bits_from_file(str(file_path))
    ext = file_path.suffix.lower()
    type_map = {'.txt': 'txt', '.tif': 'tif', '.tiff': 'tif', '.png': 'png'}
    file_type = type_map.get(ext, 'bin')

    # 24-bit type + 40-bit size = 64-bit header
    val = (ord(file_type[0]) << 16) | (ord(file_type[1]) << 8) | ord(file_type[2])
    hdr_type = np.zeros(24, dtype=np.uint8)
    for k in range(24): hdr_type[k] = (val >> (23 - k)) & 1
    hdr_size = num_to_bits_msb(len(bits), bit_num=40)
    header = np.concatenate([hdr_type, hdr_size])
    bits = np.concatenate([header, bits]).astype(np.uint8)

    # ── Scrambler ──
    if use_scrambler:
        bits = scramble_bits(bits, seed=scrambler_seed, mode='random')

    # ── LDPC encode ──
    code = ldpc_make_code(standard='802.11n', rate=rate, z=Z, ptype='A',
                          device='cuda', llr_clip=10.0, max_iter=100,
                          verbose=False, log_every=1, check_every=1)
    ldpc_bits, (K, Ncw) = ldpc_encode_bits(bits, c=code)
    if len(ldpc_bits) % 2 == 1:
        ldpc_bits = np.concatenate([ldpc_bits, np.array([0], dtype=np.uint8)])
    qpsk = QPSK_mapping(ldpc_bits.reshape(-1, 2))

    # ── OFDM modulate data ──
    data_wf = ofdm_modulate_data(qpsk)

    # ── Assemble ──
    tx = np.concatenate([chirp_front, tx_preamble, data_wf, chirp_tail])
    tx /= np.max(np.abs(tx))

    return tx, pilots_fd, {
        "pilot_mode": pilot_mode, "scrambler": use_scrambler,
        "scrambler_seed": scrambler_seed, "Z": Z, "rate": rate,
        "K": K, "Ncw": Ncw, "file_type": file_type,
        "payload_bits": int(len(bits) - 64), "tx_len": len(tx)
    }


def fname_from_meta(meta):
    scram = f"scr{meta['scrambler_seed']}" if meta['scrambler'] else "noscr"
    return (f"tx_N{N_FFT}_cp{CP_LEN}_S8{meta['pilot_mode']}"
            f"_R{meta['rate'].replace('/','-')}_Z{meta['Z']}_{scram}.npy")


# ── Test configs ──
TESTS = [
    ("S8standard, no scrambler, Z=81", "standard", False, 0, 81),
    ("S8same, no scrambler, Z=81",     "same",     False, 0, 81),
    ("S8diff, no scrambler, Z=81",     "different",False, 0, 81),
    ("S8standard, no scrambler, Z=27", "standard", False, 0, 27),
    ("S8same, random scr=256, Z=81",    "same",     True, 256, 81),
]


def main():
    original = INPUT_FILE.read_bytes()
    orig_hash = hashlib.sha256(original).hexdigest()[:16]
    print(f"Input: {INPUT_FILE}  ({len(original)} bytes, sha256={orig_hash})")
    print(f"{'='*60}")

    for desc, pilot_mode, use_scrambler, scr_seed, Z in TESTS:
        print(f"\n-- {desc} --")
        t0 = time.time()

        # TX
        tx_wf, pilots_fd, meta = build_tx(
            INPUT_FILE, pilot_mode=pilot_mode, use_scrambler=use_scrambler,
            scrambler_seed=scr_seed, Z=Z
        )
        tx_time = time.time() - t0
        fname = fname_from_meta(meta)
        np.save(SAVE_DIR / fname, tx_wf)
        print(f"  TX: {len(tx_wf)} samples ({len(tx_wf)/FS:.1f}s) -> {fname}")

        # RX
        t0 = time.time()
        args = make_args(use_scrambler=use_scrambler, scrambler_seed=scr_seed, ldpc_z=Z)
        decoded, info = receiver(tx_wf.astype(np.float64), pilots_fd, args)
        rx_time = time.time() - t0

        # decoded is ALREADY payload (receiver strips the 64-bit header)
        out_bytes = np.packbits(decoded.flatten()).tobytes()
        match = (out_bytes == original)
        out_hash = hashlib.sha256(out_bytes).hexdigest()[:16]

        suffix = meta['file_type']
        out_path = OUTPUT_DIR / f"e2e_{fname.replace('.npy','')}.{suffix}"
        with open(out_path, 'wb') as f:
            f.write(out_bytes)

        status = "OK" if match else f"MISMATCH"
        print(f"  RX: iter={info['ldpc_iter']}, payload={len(out_bytes)}B (orig={len(original)}B), "
              f"match={match}, type={info.get('type_suffix','?')}")
        print(f"  [{status}] tx={tx_time:.1f}s rx={rx_time:.1f}s -> {out_path.name}")

    # Quick summary
    print(f"\n{'='*60}")
    print("Done. Generated waveforms in save/signal/, decoded outputs in output/ldpc/")
    print(f"Original: {INPUT_FILE} ({len(original)}B)")

if __name__ == "__main__":
    main()
