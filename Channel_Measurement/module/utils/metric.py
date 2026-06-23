"""Signal analysis metrics: EVM, SNR, Es/N0, noise estimation, PAPR."""
import numpy as np


def evm_from_constellation(const: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-subcarrier EVM^2 (power-normalized), aligned on last dimension; returns same shape."""
    const = np.asarray(const)
    ref = np.asarray(ref)
    error = const - ref
    return (np.abs(error) ** 2) / (np.abs(ref) ** 2 + 1e-12)


def snr_from_constellation(const: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-subcarrier linear SNR from EVM^2, aligned on last dimension; returns same shape."""
    evm2 = evm_from_constellation(const, ref)
    return 1.0 / np.maximum(evm2, 1e-12)


def _mad_sigma(x: np.ndarray, axis=-1):
    """MAD estimate of standard deviation: sigma ~ 1.4826 * median(|x - median(x)|)."""
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=False)
    return 1.4826 * mad


def _esno_from_sigmas(sig_r: float, sig_i: float):
    """Estimate Es/N0 (linear and dB) from robust I/Q sigma estimates for QPSK (Es=1)."""
    sigma2 = 0.5 * (sig_r ** 2 + sig_i ** 2)
    esno_lin = 1.0 / (2.0 * sigma2 + 1e-12)
    esno_db = 10.0 * np.log10(esno_lin)
    return esno_lin, esno_db


def robust_sigma(constellations: np.ndarray, *, per_sc: bool = False) -> dict[str, np.ndarray]:
    """Estimate axial noise std (sigma_r, sigma_i) from QPSK hard-decision errors."""
    from .demodulate import _qpsk_hard
    s = np.asarray(constellations)
    if s.ndim == 1:
        s = s[None, :]
    hard = _qpsk_hard(s)
    err = s - hard
    if per_sc:
        sr = np.abs(err.real) * 1.2533
        si = np.abs(err.imag) * 1.2533
    else:
        sr = _mad_sigma(err.real, axis=-1) + 1e-12
        si = _mad_sigma(err.imag, axis=-1) + 1e-12
    return {"sigma_r": sr, "sigma_i": si}


def pll_snr_median(const_zf: np.ndarray) -> np.ndarray:
    """Per-symbol median SNR (dB) from ZF-equalized constellations. const_zf: [n,Nd] or [Nd]."""
    from .demodulate import _qpsk_hard
    s = np.asarray(const_zf)
    if s.ndim == 1:
        ref = _qpsk_hard(s)
        snr_sc = snr_from_constellation(s, ref)
        return 10.0 * np.log10(np.median(np.clip(snr_sc, 1e-12, None)))
    else:
        ref = _qpsk_hard(s)
        snr_sc = snr_from_constellation(s, ref)
        return 10.0 * np.log10(np.median(np.clip(snr_sc, 1e-12, None), axis=-1))


def calculate_papr(signal: np.ndarray) -> tuple[float, float]:
    """Compute Peak-to-Average Power Ratio. Returns (linear, dB)."""
    power = np.abs(signal) ** 2
    papr = np.max(power) / np.mean(power)
    return papr, 10 * np.log10(papr)
