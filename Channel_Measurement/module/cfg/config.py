# -*- coding: utf-8 -*-
"""
Unified configuration dataclasses for emitter and receiver.
Replaces scattered argparse.Namespace with typed, composable configs.

All default values match the current production receiver.py / emitter.py settings.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  Shared sub-configs
# ═══════════════════════════════════════════════════════════════

@dataclass
class OFDMConfig:
    """OFDM modulation parameters."""
    fs: int = 48000
    N: int = 8192
    cp_len: int = 1024
    num_pilot: int = 8
    data_start: int = 204
    data_tail: int = 819
    clockwise: bool = False

    @property
    def sym_len(self) -> int:
        return self.N + self.cp_len


@dataclass
class ChirpConfig:
    """Chirp synchronization signal parameters."""
    duration: float = 2.0
    f_l: int = 10
    f_h: int = 24000
    f_tail_l: int = 20
    f_tail_h: int = 18000
    two_chirp: bool = True   # True = chirp at both ends, False = front only


@dataclass
class HeaderConfig:
    """File header encoding parameters."""
    head_bit: int = 64
    size_bit_w: int = 40
    type_bit_w: int = 24
    suffix_map: dict = field(default_factory=lambda: {"tif": "tiff", "txt": "txt", "jpg": "jpg", "png": "png"})

    @property
    def suffix_map_rev(self) -> dict:
        return {v: k for k, v in self.suffix_map.items()}


@dataclass
class ScramblerConfig:
    """Scrambler parameters."""
    enabled: bool = False
    seed: int = 256
    mode: str = 'random'     # 'random' | 'LFSR'
    bitwidth: Optional[int] = None  # only for LFSR mode


@dataclass
class LDPCConfig:
    """LDPC codec parameters."""
    standard: str = '802.11n'
    rate: str = '1/2'
    z: int = 81
    ptype: str = 'A'
    device: str = 'cuda'
    batch: int = 512
    microbatch: int = 256
    llr_clip: float = 10.0
    max_iter: int = 100
    verbose: bool = False
    print_iter: bool = False
    log_every: int = 1
    check_every: int = 1


# ═══════════════════════════════════════════════════════════════
#  Receiver-specific configs
# ═══════════════════════════════════════════════════════════════

@dataclass
class PseudoPilotConfig:
    """Pseudo-pilot iterative decoding parameters."""
    max_iter: int = 20
    edge_expand: int = 32
    first_try_portion: float = 0.8
    alpha: float = 0.5       # H fusion smoothing coefficient


@dataclass
class PLLParams:
    """CPE PLL parameters (flat, for config; use to_pll_config() to get decoder_oop.PLLConfig)."""
    alpha: float = 0.15
    beta: float = 0.9
    alpha_min: float = 0.05
    alpha_max: float = 0.50
    snr_th_db: float = 6.0
    snr_scale: float = 4.0
    snr_th_min_db: float = 3.0
    snr_mid_db: float = 6.0
    snr_th_max_db: float = 20.0

    def to_pll_config(self):
        from ..utils.decoder_oop import PLLConfig
        return PLLConfig(
            alpha=self.alpha, beta=self.beta,
            alpha_min=self.alpha_min, alpha_max=self.alpha_max,
            snr_th_db=self.snr_th_db, snr_scale=self.snr_scale,
            snr_th_min_db=self.snr_th_min_db, snr_mid_db=self.snr_mid_db,
            snr_th_max_db=self.snr_th_max_db,
        )


@dataclass
class SigmaParams:
    """Noise sigma tracking parameters."""
    per_sc: bool = True
    alpha_min: float = 0.05
    alpha_max: float = 0.7
    init_sigma: float = 0.3

    def to_sigma_config(self):
        from ..utils.decoder_oop import SigmaTrackerConfig
        return SigmaTrackerConfig(
            per_sc=self.per_sc,
            alpha_min=self.alpha_min, alpha_max=self.alpha_max,
            init_sigma=self.init_sigma,
        )


@dataclass
class InterpParams:
    """Frequency offset interpolation parameters."""
    mode: str = 'hold'
    smooth: float = 0.0


@dataclass
class DisplayConfig:
    """Plot and print display control."""
    plot: bool = False
    plot_opt: dict = field(default_factory=lambda: {
        'correlation': False, 'impulse_response': False,
        'raw_pilot_constellation': False, 'corrected_pilot_constellation': False,
        'data_constellation': True, 'unwrap': False,
        'received_signal': False, 'BER_show': True,
        'snr_time_pilot': False, 'snr_time_comb': False,
        'snr_time_data': False, 'snr_over_sc': False,
        'freq_offset_interpolate': False,
    })
    print_flag: bool = True
    print_opt: dict = field(default_factory=lambda: {
        'pilot_metric': True, 'pilot_delta': False,
        'data_metric': True, 'iter_verbose': True,
    })
    print_len: int = 64
    print_pad: str = '-'


# ═══════════════════════════════════════════════════════════════
#  Top-level configs
# ═══════════════════════════════════════════════════════════════

@dataclass
class EmitterConfig:
    """Emitter configuration — all parameters needed to build a TX waveform."""
    ofdm: OFDMConfig = field(default_factory=OFDMConfig)
    chirp: ChirpConfig = field(default_factory=ChirpConfig)
    header: HeaderConfig = field(default_factory=HeaderConfig)
    scrambler: ScramblerConfig = field(default_factory=ScramblerConfig)
    ldpc: LDPCConfig = field(default_factory=LDPCConfig)
    pilot_mode: str = 'standard'   # 'standard' | 'different' | 'same'
    comb_enabled: bool = False
    comb_iter: int = 10
    comb_seed: int = 128
    fill_seed: int = 2025

    # Convenience aliases to minimize diff in emitter code
    @property
    def fs(self): return self.ofdm.fs
    @property
    def N(self): return self.ofdm.N
    @property
    def cp_len(self): return self.ofdm.cp_len
    @property
    def num_pilot(self): return self.ofdm.num_pilot
    @property
    def data_start(self): return self.ofdm.data_start
    @property
    def data_tail(self): return self.ofdm.data_tail
    @property
    def chirp_len(self): return self.chirp.duration
    @property
    def chirp_l(self): return self.chirp.f_l
    @property
    def chirp_h(self): return self.chirp.f_h
    @property
    def chirp_two(self): return self.chirp.two_chirp
    @property
    def head_bit(self): return self.header.head_bit
    @property
    def size_bit_w(self): return self.header.size_bit_w
    @property
    def type_bit_w(self): return self.header.type_bit_w
    @property
    def use_scrambler(self): return self.scrambler.enabled
    @property
    def scrambler_seed(self): return self.scrambler.seed
    @property
    def scrambler_mode(self): return self.scrambler.mode
    @property
    def ldpc_standard(self): return self.ldpc.standard
    @property
    def ldpc_rate(self): return self.ldpc.rate
    @property
    def ldpc_z(self): return self.ldpc.z
    @property
    def ldpc_ptype(self): return self.ldpc.ptype
    @property
    def use_comb(self): return self.comb_enabled


@dataclass
class ReceiverConfig:
    """Receiver configuration — all parameters needed to decode a recorded waveform."""
    ofdm: OFDMConfig = field(default_factory=OFDMConfig)
    chirp: ChirpConfig = field(default_factory=ChirpConfig)
    header: HeaderConfig = field(default_factory=HeaderConfig)
    scrambler: ScramblerConfig = field(default_factory=ScramblerConfig)
    ldpc: LDPCConfig = field(default_factory=LDPCConfig)
    pll: PLLParams = field(default_factory=PLLParams)
    sigma: SigmaParams = field(default_factory=SigmaParams)
    interp: InterpParams = field(default_factory=InterpParams)
    pseudo_pilot: PseudoPilotConfig = field(default_factory=PseudoPilotConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    comb_enabled: bool = False
    comb_interval: Optional[int] = None
    comb_pilot_seed_base: int = 128
    groundtruth: bool = False
    tx_file_path: Optional[str] = None

    # Convenience aliases matching current args.* names
    @property
    def fs(self): return self.ofdm.fs
    @property
    def N(self): return self.ofdm.N
    @property
    def cp_len(self): return self.ofdm.cp_len
    @property
    def num_pilot(self): return self.ofdm.num_pilot
    @property
    def data_start(self): return self.ofdm.data_start
    @property
    def data_tail(self): return self.ofdm.data_tail
    @property
    def clockwise(self): return self.ofdm.clockwise
    @property
    def sym_len(self): return self.ofdm.sym_len
    @property
    def chirp_len(self): return self.chirp.duration
    @property
    def chirp_l(self): return self.chirp.f_l
    @property
    def chirp_h(self): return self.chirp.f_h
    @property
    def head_bit(self): return self.header.head_bit
    @property
    def size_bit_w(self): return self.header.size_bit_w
    @property
    def type_bit_w(self): return self.header.type_bit_w
    @property
    def suffix_map(self): return self.header.suffix_map
    @property
    def suffix_map_rev(self): return self.header.suffix_map_rev
    @property
    def use_scrambler(self): return self.scrambler.enabled
    @property
    def scrambler_seed(self): return self.scrambler.seed
    @property
    def scrambler_mode(self): return self.scrambler.mode
    @property
    def scrambler_bitwidth(self): return self.scrambler.bitwidth
    @property
    def ldpc_standard(self): return self.ldpc.standard
    @property
    def ldpc_rate(self): return self.ldpc.rate
    @property
    def ldpc_z(self): return self.ldpc.z
    @property
    def ldpc_ptype(self): return self.ldpc.ptype
    @property
    def ldpc_device(self): return self.ldpc.device
    @property
    def ldpc_batch(self): return self.ldpc.batch
    @property
    def ldpc_microbatch(self): return self.ldpc.microbatch
    @property
    def ldpc_llr_clip(self): return self.ldpc.llr_clip
    @property
    def ldpc_max_iter(self): return self.ldpc.max_iter
    @property
    def ldpc_verbose(self): return self.ldpc.verbose
    @property
    def ldpc_print_iter(self): return self.ldpc.print_iter
    @property
    def ldpc_log_every(self): return self.ldpc.log_every
    @property
    def ldpc_check_every(self): return self.ldpc.check_every
    @property
    def pll_alpha(self): return self.pll.alpha
    @property
    def pll_beta(self): return self.pll.beta
    @property
    def pll_alpha_min(self): return self.pll.alpha_min
    @property
    def pll_alpha_max(self): return self.pll.alpha_max
    @property
    def pll_snr_th_db(self): return self.pll.snr_th_db
    @property
    def pll_snr_scale(self): return self.pll.snr_scale
    @property
    def pll_snr_th_min_db(self): return self.pll.snr_th_min_db
    @property
    def pll_snr_min_db(self): return self.pll.snr_th_min_db  # legacy alias
    @property
    def pll_snr_mid_db(self): return self.pll.snr_mid_db
    @property
    def pll_snr_th_max_db(self): return self.pll.snr_th_max_db
    @property
    def pll_snr_max_db(self): return self.pll.snr_th_max_db  # legacy alias
    @property
    def sig_trk_per_sc(self): return self.sigma.per_sc
    @property
    def sig_trk_alpha_min(self): return self.sigma.alpha_min
    @property
    def sig_trk_alpha_max(self): return self.sigma.alpha_max
    @property
    def sig_trk_init_sigma(self): return self.sigma.init_sigma
    @property
    def interp_mode(self): return self.interp.mode
    @property
    def interp_smooth(self): return self.interp.smooth
    @property
    def plot(self): return self.display.plot
    @property
    def plot_opt(self): return self.display.plot_opt
    @property
    def print_flag(self): return self.display.print_flag
    @property
    def print_opt(self): return self.display.print_opt
    @property
    def print_len(self): return self.display.print_len
    @property
    def print_pad(self): return self.display.print_pad
    @property
    def use_comb(self): return self.comb_enabled
    @property
    def INTERVAL(self): return self.comb_interval
    @property
    def COMB_PILOT_SEED_BASE(self): return self.comb_pilot_seed_base
    @property
    def edge_expand(self): return self.pseudo_pilot.edge_expand
    @property
    def max_pseudo_iter(self): return self.pseudo_pilot.max_iter
    @property
    def first_try_portion(self): return self.pseudo_pilot.first_try_portion
    @property
    def pseudo_pilot_alpha(self): return self.pseudo_pilot.alpha
    @property
    def pseudo_pilot_max_iter(self): return self.pseudo_pilot.max_iter

    def to_pll_config(self):
        return self.pll.to_pll_config()

    def to_sigma_config(self):
        return self.sigma.to_sigma_config()
