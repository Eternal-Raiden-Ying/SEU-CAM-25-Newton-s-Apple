# -*- coding: utf-8 -*-
"""
Reusable emitter function — builds a complete OFDM+LDPC transmit waveform.
Follows the same pattern as receiver_stable.receiver().
"""
from __future__ import annotations
import numpy as np
import argparse
from pathlib import Path

from ..utils.modulate import (
    generate_chirp, QPSK_mapping, OFDM_modulate_data,
    OFDM_modulate_data_with_comb, ofdm_modulate_symbol,
)
from ..utils.encode import ldpc_encode_bits, scramble_bits
from ..utils.io_interface import get_bits_from_file, num_to_bits_msb, ascii3_to_24bits
from ..utils.batch import generate_pilot_symbol


def emitter(
    file_path: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build a complete OFDM+LDPC transmit waveform.

    Parameters:
        file_path: path to the file to transmit (txt, tiff, png, ...)
        args: argparse.Namespace with these relevant fields:
            fs, N, cp_len         — OFDM params
            num_pilot             — number of pilot symbols
            pilot_mode            — "different" | "same" | "standard"
            chirp_len, chirp_l, chirp_h  — chirp params
            head_bit, size_bit_w, type_bit_w  — header config
            use_scrambler, scrambler_seed, scrambler_mode  — scrambler
            ldpc_standard, ldpc_rate, ldpc_z, ldpc_ptype  — LDPC
            use_comb, comb_iter, comb_seed  — comb pilot config
            data_start, data_tail, fill_seed  — OFDM guard band config

    Returns:
        tx_signal  : 1D real time-domain waveform (float64)
        pilots_fd  : [num_pilot, N] frequency-domain pilot reference
        meta       : dict with K, Ncw, file_type, etc.
    """
    fs = args.fs
    N = args.N
    cp_len = args.cp_len
    num_pilot = args.num_pilot
    pilot_mode = getattr(args, "pilot_mode", "standard")

    # ── Chirps ──
    chirp_front = generate_chirp(fs, duration=args.chirp_len,
                                 f_l=args.chirp_l, f_h=args.chirp_h) * 0.4
    chirp_tail  = generate_chirp(fs, duration=args.chirp_len,
                                 f_l=20, f_h=18000) * 0.4

    # ── Pilot preamble ──
    if pilot_mode == "standard":
        # Try loading pre-saved standard pilot; fallback to "different"
        std_path = Path(__file__).resolve().parents[3] / "save" / "pilot" / "pilot_STANDARD_freq_domain.npy"
        if std_path.exists():
            pilots_fd = np.load(std_path)
        else:
            pilots_fd = np.array([generate_pilot_symbol(N, 256 + i) for i in range(num_pilot)])
    elif pilot_mode == "different":
        pilots_fd = np.array([generate_pilot_symbol(N, 256 + i) for i in range(num_pilot)])
    else:  # "same"
        p = generate_pilot_symbol(N, 256)
        pilots_fd = np.array([p] * num_pilot)

    tx_preamble = np.concatenate([ofdm_modulate_symbol(p, cp_len) for p in pilots_fd])
    tx_preamble /= np.max(np.abs(tx_preamble))

    # ── Chirp placement: [chirp_front, pilots] ──
    chirp_two = getattr(args, "chirp_two", True)
    tx_preamble = np.concatenate([chirp_front, tx_preamble])

    # ── Read file + 64-bit header ──
    bits = get_bits_from_file(file_path)
    ext = Path(file_path).suffix.lower()
    type_map = {'.txt': 'txt', '.tif': 'tif', '.tiff': 'tif', '.png': 'png'}
    file_type = type_map.get(ext, 'bin')

    hdr_type = ascii3_to_24bits(file_type)
    hdr_size = num_to_bits_msb(len(bits), bit_num=args.size_bit_w)
    header = np.concatenate([hdr_type, hdr_size])
    bits = np.concatenate([header, bits]).astype(np.uint8)

    # ── Scrambler ──
    if getattr(args, "use_scrambler", False):
        bits = scramble_bits(bits, seed=args.scrambler_seed, mode=args.scrambler_mode)

    # ── LDPC encode ──
    from ..utils.encode import ldpc_make_code
    code = ldpc_make_code(
        standard=args.ldpc_standard, rate=args.ldpc_rate,
        z=args.ldpc_z, ptype=args.ldpc_ptype,
        device='cuda', llr_clip=10.0, max_iter=100,
        verbose=False, log_every=1, check_every=1,
    )
    ldpc_bits, (K, Ncw) = ldpc_encode_bits(bits, c=code)
    if len(ldpc_bits) % 2 == 1:
        ldpc_bits = np.concatenate([ldpc_bits, np.array([0], dtype=np.uint8)])
    qpsk = QPSK_mapping(ldpc_bits.reshape(-1, 2))

    # ── OFDM modulate ──
    data_start = getattr(args, "data_start", 204)
    data_tail  = getattr(args, "data_tail", 819)
    fill_seed  = getattr(args, "fill_seed", 2025)

    if getattr(args, "use_comb", False):
        comb_iter = getattr(args, "comb_iter", 10)
        comb_seed = getattr(args, "comb_seed", 128)
        data_wf, _ = OFDM_modulate_data_with_comb(
            qpsk, N, cp_len,
            iteration=comb_iter, seed=comb_seed,
            data_start=data_start, data_tail=data_tail, fill_seed=fill_seed,
        )
    else:
        data_wf = OFDM_modulate_data(
            qpsk, N, cp_len,
            data_start=data_start, data_tail=data_tail, fill_seed=fill_seed,
        )

    # ── Assemble final signal ──
    if chirp_two:
        tx = np.concatenate([tx_preamble, data_wf, chirp_tail])
    else:
        tx = np.concatenate([tx_preamble, data_wf])
    tx /= np.max(np.abs(tx))

    meta = {
        "K": K, "Ncw": Ncw, "Z": args.ldpc_z, "rate": args.ldpc_rate,
        "pilot_mode": pilot_mode, "scrambler": getattr(args, "use_scrambler", False),
        "scrambler_seed": getattr(args, "scrambler_seed", 0),
        "comb": getattr(args, "use_comb", False),
        "file_type": file_type, "payload_bits": int(len(bits) - args.head_bit),
        "tx_len": len(tx), "N": N, "cp_len": cp_len,
    }
    return tx, pilots_fd, meta
