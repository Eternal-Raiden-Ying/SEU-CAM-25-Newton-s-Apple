import numpy as np


def phase_unwrap(data:np.ndarray, estimate_start=0, estimate_percent=0.1,*,
                 initial_discont=1.5*np.pi, mode='truncated', amp_filter_th=1.2):
    """
        improved unwrap phase function
        first use default unwrap() to get a linear part and unwrap precisely based on that,
        so make sure arguments are set right to get the initial linear part,
        related arg: estimate_start, estimate_percent, initial_discont
    :param data: phase unwrapped (not only the angle, amplitude will be used to filter)
    :param estimate_start: start from where you get the linear part
    :param estimate_percent: length percent of data to estimate the linear part
    :param initial_discont: discont for default unwrap(), the larger discont is, the more func will tolerant noise
    :param x_filtered: if data is filtered, raw x val is needed to do polyfit
    :param amp_filter_th: Amplitude filter threshold, the more it nearer to 1, the more strict the filter is
    :param mode: unwrap mode, supported ['simple', 'filtered', 'truncated']
    :return: index, unwrapped phase
    """

    if mode == 'simple':
        x = np.arange(data.size)
        y = np.angle(data)
    elif mode == 'filtered':
        mask = np.where(np.abs(data) < amp_filter_th)
        x = np.arange(data.size)[mask]
        y = np.angle(data)[mask]
    elif mode == 'truncated':
        N = data.size
        data = data[:N//2]
        mask = np.where(np.abs(data) < amp_filter_th)
        x = np.arange(data.size)[mask]
        y = np.angle(data)[mask]
    else:
        raise ValueError("Unexpected mode, supported ['simple', 'filtered', 'truncated']")

    initial_res = np.unwrap(y, discont=initial_discont)
    estimate_len = int(y.size * estimate_percent)
    estimate_part = initial_res[estimate_start:estimate_start+estimate_len]
    coeffs = np.polyfit(x[estimate_start:estimate_start+estimate_len], estimate_part, deg=1)
    slope, intercept = coeffs
    fit_line = slope * x + intercept
    unwrap_refined = data + 2 * np.pi * np.round((fit_line - data) / (2 * np.pi))
    return x, unwrap_refined

def fitting_line(x, y,*, filter=True, residual_th=2):
    """
        fit a line for given x and y (y=kx+b), then return k,b
        package the residual filter, thus fitting could be more precise
    :param x:
    :param y:
    :param filter: boolean
    :param residual_th: use for mask, threshold = std * residual_th
    :return:
    """
    coeffs = np.polyfit(x,y,deg=1)
    if filter:
        fit_line = np.polyval(coeffs, x)
        residual = y - fit_line
        std = np.std(residual)
        mask = np.where(np.abs(residual) < residual_th * std)
        coeffs_refined = np.polyfit(x[mask], y[mask], deg=1)
        slope, intercept = coeffs_refined
    else:
        slope, intercept = coeffs

    return slope, intercept


def smooth_H_moving_average(H_f, window_size=11):
    """
        few use
    :param H_f:
    :param window_size:
    :return:
    """
    window = np.ones(window_size) / window_size
    H_real = np.convolve(H_f.real, window, mode='same')
    H_imag = np.convolve(H_f.imag, window, mode='same')
    return H_real + 1j * H_imag


def normalize(data: np.ndarray,*, axis=-1, keepdim=False):
    """
        normalize given data
    :param data:
    :return:
    """
    if data.ndim == 1:
        max_val = np.max(np.abs(data))
    else:
        max_val = np.max(np.abs(data), axis=axis, keepdims=keepdim)
    return data / max_val


if __name__ == "__main__":
    # here you can test these function if you are not familiar with them
    print()
