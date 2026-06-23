# -*- coding: utf-8 -*-
"""
Reusable emitter function — builds a complete OFDM+LDPC transmit waveform.
Accepts an EmitterConfig dataclass (module.cfg.config).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from ..cfg.config import EmitterConfig
from ..utils.modulate import (
    generate_chirp, QPSK_mapping, OFDM_modulate_data,
    OFDM_modulate_data_with_comb, ofdm_modulate_symbol,
    generate_pilot_symbol,
)
from ..utils.encode import ldpc_encode_bits, scramble_bits
from ..utils.io_interface import get_bits_from_file, num_to_bits_msb, ascii3_to_24bits


def emitter(
    file_path: str,
    cfg: EmitterConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build a complete OFDM+LDPC transmit waveform.

    Parameters:
        file_path: path to the file to transmit (txt, tiff, png, ...)
        cfg: EmitterConfig dataclass

    Returns:
        tx_signal  : 1D real time-domain waveform (float64)
        pilots_fd  : [num_pilot, N] frequency-domain pilot reference
        meta       : dict with K, Ncw, file_type, etc.
    """
    fs = cfg.fs
    N = cfg.N
    cp_len = cfg.cp_len
    num_pilot = cfg.num_pilot

    # ── Chirps ──
    chirp_front = generate_chirp(fs, duration=cfg.chirp_len,
                                 f_l=cfg.chirp_l, f_h=cfg.chirp_h) * 0.4
    chirp_tail  = generate_chirp(fs, duration=cfg.chirp_len,
                                 f_l=cfg.chirp.f_tail_l, f_h=cfg.chirp.f_tail_h) * 0.4

    # ── Pilot preamble ──
    pilot_mode = cfg.pilot_mode
    if pilot_mode == "standard":
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
    tx_preamble = np.concatenate([chirp_front, tx_preamble])

    # ── Read file + 64-bit header ──
    bits = get_bits_from_file(file_path)
    ext = Path(file_path).suffix.lower()
    type_map = {'.txt': 'txt', '.tif': 'tif', '.tiff': 'tif', '.png': 'png'}
    file_type = type_map.get(ext, 'bin')

    hdr_type = ascii3_to_24bits(file_type)
    hdr_size = num_to_bits_msb(len(bits), bit_num=cfg.size_bit_w)
    header = np.concatenate([hdr_type, hdr_size])
    bits = np.concatenate([header, bits]).astype(np.uint8)

    # ── Scrambler ──
    if cfg.use_scrambler:
        bits = scramble_bits(bits, seed=cfg.scrambler_seed, mode=cfg.scrambler_mode)

    # ── LDPC encode ──
    from ..utils.encode import ldpc_make_code
    code = ldpc_make_code(
        standard=cfg.ldpc_standard, rate=cfg.ldpc_rate,
        z=cfg.ldpc_z, ptype=cfg.ldpc_ptype,
        device=cfg.ldpc.device, llr_clip=cfg.ldpc.llr_clip, max_iter=cfg.ldpc.max_iter,
        verbose=cfg.ldpc.verbose, log_every=cfg.ldpc.log_every, check_every=cfg.ldpc.check_every,
    )
    ldpc_bits, (K, Ncw) = ldpc_encode_bits(bits, c=code)
    if len(ldpc_bits) % 2 == 1:
        ldpc_bits = np.concatenate([ldpc_bits, np.array([0], dtype=np.uint8)])
    qpsk = QPSK_mapping(ldpc_bits.reshape(-1, 2))

    # ── OFDM modulate ──
    if cfg.use_comb:
        data_wf, _ = OFDM_modulate_data_with_comb(
            qpsk, N, cp_len,
            iteration=cfg.comb_iter, seed=cfg.comb_seed,
            data_start=cfg.data_start, data_tail=cfg.data_tail, fill_seed=cfg.fill_seed,
        )
    else:
        data_wf = OFDM_modulate_data(
            qpsk, N, cp_len,
            data_start=cfg.data_start, data_tail=cfg.data_tail, fill_seed=cfg.fill_seed,
        )

    # ── Assemble final signal ──
    if cfg.chirp_two:
        tx = np.concatenate([tx_preamble, data_wf, chirp_tail])
    else:
        tx = np.concatenate([tx_preamble, data_wf])
    tx /= np.max(np.abs(tx))

    meta = {
        "K": K, "Ncw": Ncw, "Z": cfg.ldpc_z, "rate": cfg.ldpc_rate,
        "pilot_mode": pilot_mode, "scrambler": cfg.use_scrambler,
        "scrambler_seed": cfg.scrambler_seed,
        "comb": cfg.use_comb,
        "file_type": file_type, "payload_bits": int(len(bits) - cfg.head_bit),
        "tx_len": len(tx), "N": N, "cp_len": cp_len,
    }
    return tx, pilots_fd, meta
