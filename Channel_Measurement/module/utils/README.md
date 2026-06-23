# module/utils

Utility functions for the OFDM acoustic communication pipeline.

## File Index

### Modulation — `modulate.py`
| Function | Purpose |
|----------|---------|
| `generate_chirp()` | Linear chirp signal (scipy.signal.chirp wrapper) |
| `QPSK_mapping()` | Bits → QPSK complex symbols |
| `OFDM_modulate_data()` | QPSK symbols → OFDM time-domain waveform |
| `OFDM_modulate_data_with_comb()` | OFDM modulation with embedded comb pilots |
| `ofdm_modulate_symbol()` | Single OFDM symbol: IFFT + CP |
| `generate_pilot_symbol()` | Frequency-domain pilot with known QPSK sequence |
| `generate_comb_pilot_symbol()` | Comb pilot symbol for in-band channel tracking |
| `calculate_papr()` | Peak-to-Average Power Ratio (in `metric.py`) |

### Demodulation & Equalization — `demodulate.py`
| Function | Purpose |
|----------|---------|
| `get_symbols()` | Strip cyclic prefix from time-domain OFDM symbols |
| `get_constellation()` | Zero-forcing equalization: Y(f) / H(f) |
| `_qpsk_hard()` | Hard QPSK decision |
| `QPSK_reflection()` | Constellation points → bit pairs |
| `mmse_shrinkage()` | MMSE shrinkage estimator |
| `get_bytes()` | Bits → byte array |

### Encoding — `encode.py`
| Function | Purpose |
|----------|---------|
| `ldpc_encode_bits()` | LDPC encode with zero-padding to codeword boundary |
| `ldpc_make_code()` | Build LDPC code object (802.11n / 802.16e) |
| `scramble_bits()` | Scramble bit sequence (random or LFSR) |
| `scrambler()` | LFSR-based scrambler |
| `scrambler_random()` | PRNG-based scrambler |

### Decoding — `decode.py`
| Function | Purpose |
|----------|---------|
| `ldpc_decode_blocks()` | Batch LDPC decode (GPU via DGL, CPU fallback) |
| `llr_from_constellation()` | QPSK constellation → approximate LLR |
| `llr_scale_by_snr()` | Scale LLRs by per-subcarrier SNR |
| `pack_llr_blocks()` | Pack LLRs into codeword-sized blocks |

### Soft Decoder — `decoder_oop.py`
Object-oriented receiver with iterative decoding.

| Class | Purpose |
|-------|---------|
| `OFDMSoftDecoder` | OOP receiver with pseudo-pilot iterative decode |
| `DD_CPE_PLL` | Decision-directed common phase error PLL |
| `SigmaTracker` | Per-subcarrier noise variance tracking |
| `DriftGuardConfig` | Anomaly detection for SFO/CFO drift |

### Channel Estimation — `channel_estimate.py`
| Function | Purpose |
|----------|---------|
| `evaluate_H_f()` | Estimate H(f) from received pilots (1D or 2D) |
| `correct_H_f()` | Apply linear phase rotation (SFO/CFO correction) |
| `estimate_drift_and_origin()` | Fit delta & phi_step from preamble pilot sequence |
| `_fit_drift_between()` | Fit drift between two H(f) snapshots |
| `_pilot_quality()` | Evaluate comb pilot quality via SNR |
| `build_segments_from_pilots()` | Build per-symbol channel references with segment fusion |
| `analyze_pilots()` | Vectorized pilot SNR/BER/SER analysis |
| `estimate_M_from_filesize()` | Predict total OFDM symbol count from file size |
| `choose_next_pilots()` | Select comb pilot positions for next iteration |

### Metrics — `metric.py`
| Function | Purpose |
|----------|---------|
| `snr_from_constellation()` | Per-subcarrier SNR from constellation dispersion |
| `evm_from_constellation()` | Error Vector Magnitude |
| `_mad_sigma()` | Robust noise std (MAD estimator) |
| `_esno_from_sigmas()` | Es/N0 from real/imag sigma |
| `calculate_papr()` | Peak-to-Average Power Ratio |

### Synchronization — `synchronize.py`
| Function | Purpose |
|----------|---------|
| `synchronize()` | Cross-correlation chirp sync via `scipy.signal.correlate` |

### I/O — `io_interface.py`
| Function | Purpose |
|----------|---------|
| `get_bits_from_file()` | Read file → bit array (txt, tiff, png, bin) |
| `get_bits_from_str()` | String → bit array |
| `num_to_bits_msb()` | Integer → MSB-first bit vector |
| `ascii3_to_24bits()` | 3-char ASCII string → 24-bit vector |
| `emitter_cfg_to_fname()` | EmitterConfig → descriptive filename |
| `fname_to_emitter_cfg()` | Parse filename → EmitterConfig |

### Math & Signal Processing
| File | Contents |
|------|----------|
| `math_process.py` | Phase unwrapping, line fitting, normalization, segment means |
| `sfo_process.py` | SFO/CFO drift fitting and segment construction |
| `DeltaInterpolator.py` | `DeltaInterpolator` class — hold / linear / cubic drift interpolation |

### Plotting — `plot.py`
Constellation diagrams, correlation traces, SNR heatmaps, impulse response, decoding process visualization.

### Recording — `record.py`
| Function | Purpose |
|----------|---------|
| `record_signal()` | Record audio via `sounddevice` |
| `record_signal_with_error()` | Record with error handling |

### LDPC Codec — `ldpc_jossy/`
Third-party LDPC library implementing IEEE 802.11n and 802.16e.
- `py/ldpc.py` — Core encoder/decoder (also compiled to C: `bin/c_ldpc.dll`)
- Supports GPU-accelerated sum-product decoding via DGL
