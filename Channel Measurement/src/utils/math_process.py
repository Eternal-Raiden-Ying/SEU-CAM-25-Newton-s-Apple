import numpy as np


def phase_unwrap(data:np.ndarray, estimate_start=0, estimate_percent=0.1,*,
                 initial_discont=1.5*np.pi, x_filtered=None):
    if x_filtered is not None:
        x = x_filtered
    else:
        x = np.arange(data.size)
    initial_res = np.unwrap(data, discont=initial_discont)
    estimate_len = int(data.size * estimate_percent)
    estimate_part = initial_res[estimate_start:estimate_start+estimate_len]
    coeffs = np.polyfit(x[estimate_start:estimate_start+estimate_len], estimate_part, deg=1)
    slope, intercept = coeffs
    fit_line = slope * x + intercept
    unwrap_refined = data + 2 * np.pi * np.round((fit_line - data) / (2 * np.pi))
    return unwrap_refined


def smooth_H_moving_average(H_f, window_size=11):
    window = np.ones(window_size) / window_size
    H_real = np.convolve(H_f.real, window, mode='same')
    H_imag = np.convolve(H_f.imag, window, mode='same')
    return H_real + 1j * H_imag
