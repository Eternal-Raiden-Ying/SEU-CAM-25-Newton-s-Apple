"""Signal analysis metrics."""
import numpy as np


def calculate_papr(signal: np.ndarray) -> tuple[float, float]:
    """
    Compute Peak-to-Average Power Ratio of a time-domain signal.
    Returns (papr_linear, papr_dB).
    """
    power = np.abs(signal) ** 2
    papr = np.max(power) / np.mean(power)
    return papr, 10 * np.log10(papr)
