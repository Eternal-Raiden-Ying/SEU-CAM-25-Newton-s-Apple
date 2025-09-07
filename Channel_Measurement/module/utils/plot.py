import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from .math_process import normalize
from .demodulate import get_constellation
from .batch import correct_H_f

# 设置字体对象
ch_font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')   # 中文（Mac 示例）
en_font = FontProperties(family='DejaVu Sans')                               # 英文

def draw_in_TD(time, signal: np.ndarray,*,
               title: str = 'signal in time domain',
               ax: Axes = None,
               x_label: str = "",
               y_label: str = ""):
    """
        given signal duration and signal in time domain, draw it in TD
    :param time: signal duration
    :param signal: signal in TD
    :param title: pic title
    :param ax: Axes, for subplots
    :param x_label:
    :param y_label:
    :return: None
    """

    if isinstance(time, int) or isinstance(time, float):
        x = np.linspace(0, time, signal.size)
    elif isinstance(time, np.ndarray):
        x = time
    else:
        raise TypeError(f"Unsupported type, expected num or np.ndarray, received {type(time)}")

    y = signal.flatten()

    if ax is None:
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()
        plt.show()
    else:
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)


def draw_in_FD(freq, signal: np.ndarray,*,
               title: str = 'signal in frequency domain',
               ax: Axes = None,
               x_label: str = "",
               y_label: str = "",
               half: bool = True,
               mode: str = 'Amplitude',
               ignore_zero: bool = True,
               eps: float = 1e-12):
    """
        given fs and signal in TD, draw the freq pic
    :param freq:  sampling freq
    :param signal: signal in TD !!!
    :param title: pic title
    :param ax: Axes, for subplots
    :param x_label:
    :param y_label:
    :param half: boolean, show only in positive freq
    :param mode: supported ['Amplitude', 'Phase']
    :param ignore_zero: boolean, for mode 'Amplitude' if encounter 0 ignore it to get a smoother line
    :param eps:
    :return: None
    """

    freq_shift = False
    if isinstance(freq, int) or isinstance(freq, float):
        freq_shift = True
        if half:
            x = np.linspace(0, freq/2, signal.size//2)
        else:
            x = np.linspace(-freq/2, freq/2, signal.size)
    elif isinstance(freq, np.ndarray):
        x = freq
    else:
        raise TypeError(f"Unsupported type, expected num or np.ndarray, received {type(freq)}")

    if mode == 'Amplitude':
        mag = np.abs(fft(signal)).flatten()
        y = 20 * np.log10(np.where(mag == 0, eps, mag))
    elif mode == 'Phase':
        y = np.angle(fft(signal)).flatten()
    else:
        raise ValueError(f"valid mode in ['Amplitude', 'Phase'], got {mode} yet")

    if freq_shift:
        neg_freq = y[y.size//2:]
        pos_freq = y[:y.size//2]
        if half:
            y = pos_freq
        else:
            y = np.concatenate([neg_freq, pos_freq])

    if mode == "Amplitude" and ignore_zero:
        mask = np.where(y > 20 * np.log10(eps))
        x = x[mask]
        y = y[mask]

    if ax is None:
        plt.scatter(x, y, s=1, alpha=0.5)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label + ' (dB)') if 'dB' not in y_label and mode == 'Amplitude' else plt.ylabel(y_label)
        plt.grid()
        plt.show()
    else:
        ax.scatter(x, y, s=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label + ' (dB)') if 'dB' not in y_label and mode == 'Amplitude' else plt.ylabel(y_label)
        ax.grid(True)


def draw_constellation_map(received, emit_pilot, mode='QPSK',
                           title='constellation_map',*,
                           ax=None,pth=None, filename=None,
                           limit_border=5):
    """

    :param received: received signal in freq domain, without the conjugate part
    :param emit_pilot: emit pilot signal without the conjugate part
    :param ax:  Axes, for subplots
    :param title:  title for pic
    :param pth: if the sig needs to save, specified
    :param filename: if the sig needs to save, specified
    :return: None
    """
    constellation_emit = normalize(emit_pilot).flatten()
    constellation = received.flatten()
    judge_radius = 0.1

    if mode == 'QPSK':
        red_mask = np.where(np.abs(constellation_emit-(1+1j)/np.sqrt(2)) < judge_radius)
        green_mask = np.where(np.abs(constellation_emit-(-1+1j)/np.sqrt(2)) < judge_radius)
        blue_mask = np.where(np.abs(constellation_emit-(-1-1j)/np.sqrt(2)) < judge_radius)
        yellow_mask = np.where(np.abs(constellation_emit-(1-1j)/np.sqrt(2)) < judge_radius)

        real = np.real(constellation)
        imag = np.imag(constellation)
        groups = {
            'RED': {'real': real[red_mask], 'imag': imag[red_mask], 'color': 'red', 'label': '1+j'},
            'GREEN': {'real': real[green_mask], 'imag': imag[green_mask], 'color': 'green', 'label': '-1+j'},
            'BLUE': {'real': real[blue_mask], 'imag': imag[blue_mask], 'color': 'blue', 'label': '-1-j'},
            'YELLOW': {'real': real[yellow_mask], 'imag': imag[yellow_mask], 'color': 'yellow', 'label': '1-j'}
        }
        if ax:
            ax.set_xlim(-limit_border, limit_border)
            ax.set_ylim(-limit_border, limit_border)
            ax.set_title(title)
            for k, data in groups.items():
                ax.scatter(data['real'], data['imag'], c=data['color'], label=data['label'], alpha=0.6, s=1)
            ax.grid(True)
        else:
            plt.title(title)
            for k, data in groups.items():
                plt.scatter(data['real'], data['imag'], c=data['color'], label=data['label'], alpha=0.6, s=1)
            plt.grid()
            plt.xlim(-limit_border,limit_border)
            plt.ylim(-limit_border,limit_border)

        if filename:
            plt.savefig(os.path.join(pth, filename), dpi=300) if pth else plt.savefig(filename, dpi=300)

    else:
        raise ValueError("Unsupported mode, mode should be in ['QPSK',]")


def auto_constellation_map_param(num_sym):
    if num_sym <= 4:
        row = 1
        col = num_sym
        figsize = (3*num_sym,4)
    elif 4<num_sym<=8:
        row = 2
        col = 4
        figsize = (10,6)
    elif 8<num_sym<=12:
        row = 3
        col = 4
        figsize = (9,8)
    else:
        row = 4
        col = 4
        figsize = (9,10)
    return row, col, figsize


def plot_correlation(corr, *, axvline_dict=None):
    plt.plot(corr, color='royalblue', alpha=0.5)
    if axvline_dict is not None:
        for k, v in axvline_dict.items():
            plt.axvline(v, linestyle='dotted', color='red', alpha=0.5, label=k)
        plt.legend()
    plt.show()

def plot_impulse_response(h_t, fs,*,freq_half=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    draw_in_TD(time=h_t.size / fs, signal=h_t, title='Impulse response in time domain',
               ax=axes[0], x_label='time/s', y_label='h(t)')
    draw_in_FD(freq=fs, signal=h_t, title='Impulse response in freq domain', half=freq_half,
               ax=axes[1], mode='Amplitude', y_label='H(f)/dB', x_label='Freq/Hz')
    plt.show()

def plot_unwrap_phase_fitting(phase_shift, slope, intercept, x_auto, auto_unwrapped_phase, N):
    num_sym = phase_shift.shape[0] if phase_shift.ndim > 1 else 1
    n_rows, n_cols, figsize = auto_constellation_map_param(num_sym)
    if n_cols * n_cols == 1:
        x = np.linspace(-N // 2, N // 2, N, endpoint=False)
        phase_shift = np.concatenate([phase_shift[N // 2:], phase_shift[:N // 2]])
        plt.title(f"Unwrap phase fitting line")
        plt.plot(x, np.angle(phase_shift), color='orange', label='original', alpha=0.5)
        plt.plot(x, slope * x + intercept, linestyle='solid', label='fitting result', color='red')
        plt.plot(x, slope * x + intercept + np.pi, linestyle='dotted', color='red', alpha=0.5)
        plt.plot(x, slope * x + intercept - np.pi, linestyle='dotted', color='red', alpha=0.5)
        plt.scatter(x_auto, auto_unwrapped_phase, label='auto_unwrap', s=1, color='green', marker='*', alpha=0.5)
        plt.axhline(0, linestyle='dotted', color='black', linewidth=2)
        plt.axvline(0, linestyle='dotted', color='black', linewidth=2)
        plt.xlabel("sampling point")
        plt.ylabel("unwrapped phase")
        plt.legend(loc='lower right')
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*2, figsize[1]))
        for i in range(num_sym):
            ax = axes[i // n_cols, i % n_cols]
            x = np.linspace(-N // 2, N // 2, N, endpoint=False)
            phase = np.concatenate([phase_shift[i, N//2:],phase_shift[i, :N//2]])
            ax.set_title(f"Unwrap phase fitting line {i+1}")
            ax.plot(x, np.angle(phase), color='orange', label='original', alpha=0.5)
            ax.plot(x, slope[i] * x + intercept[i], linestyle='solid', label='fitting result', color='red')
            ax.plot(x, slope[i] * x + intercept[i] + np.pi, linestyle='dotted', color='red', alpha=0.5)
            ax.plot(x, slope[i] * x + intercept[i] - np.pi, linestyle='dotted', color='red', alpha=0.5)
            ax.scatter(x_auto[i], auto_unwrapped_phase[i], label='auto_unwrap', s=1, color='green', marker='*', alpha=0.5)
            ax.axhline(0, linestyle='dotted', color='black', linewidth=2)
            ax.axvline(0, linestyle='dotted', color='black', linewidth=2)
            ax.set_xlabel("sampling point")
            ax.set_ylabel("unwrapped phase")
            ax.legend(loc='lower right')
        fig.tight_layout()
    plt.show()

def plot_original_constellations(symbols_td, H_used, pilot, *, DATA_BINS=None, pic_idx=None):
    num_sym, N = symbols_td.shape
    assert symbols_td.shape == pilot.shape, "pilot should have the same shape with symbols_td"
    if pic_idx is None:
        pic_idx = np.arange(num_sym)
    if DATA_BINS is None:
        DATA_BINS = np.arange(N)

    pic_num = pic_idx.size
    n_rows, n_cols, figsize = auto_constellation_map_param(pic_num)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, index in enumerate(pic_idx):
        raw_constellation = get_constellation(symbols_td=symbols_td[index, :],
                                              H_used=H_used,
                                              DATA_BINS=DATA_BINS)
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=raw_constellation, emit_pilot=pilot[index, DATA_BINS], ax=ax,
                               title=f"constellation{index + 1}",limit_border=2)
    fig.suptitle("original constellation")
    plt.tight_layout()
    plt.show()

def plot_corrected_constellations(symbols_td, origin_H_f, pilot, symbol_len, delta,
                                  fixed_phase_shift_factor, *, DATA_BINS=None, pic_idx=None):
    num_sym, N = symbols_td.shape
    assert symbols_td.shape == pilot.shape, "pilot should have the same shape with symbols_td"
    if pic_idx is None:
        pic_idx = np.arange(num_sym)
    if DATA_BINS is None:
        DATA_BINS = np.arange(N)

    pic_num = pic_idx.size
    n_rows, n_cols, figsize = auto_constellation_map_param(pic_num)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, index in enumerate(pic_idx):
        corrected_H_f = correct_H_f(origin_H_f, N, index, symbol_len, delta, fixed_phase_shift_factor)
        corrected_constellation = get_constellation(symbols_td=symbols_td[index, :],
                                                    H_used=corrected_H_f,
                                                    DATA_BINS=DATA_BINS)
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=corrected_constellation, emit_pilot=pilot[index, DATA_BINS], ax=ax,
                               title=f"constellation{index + 1}",limit_border=2)
    fig.suptitle("corrected constellation")
    plt.tight_layout()
    plt.show()

def plot_received_signal(rx, ofdm_start, num_symbols, N, cp_len, M):
    plt.plot(rx)
    plt.axvline(ofdm_start, linestyle='dotted', color='red')
    plt.axvline(ofdm_start + (num_symbols+M) * (N+cp_len), linestyle='dotted', color='red')
    plt.axvline(ofdm_start + num_symbols * (N + cp_len), linestyle='dotted', color='red')
    plt.show()

def plot_data_constellations(const, const_ref, *, data_pos=None, pic_idx=None):
    num_const, N = const.shape
    assert const_ref.shape == const.shape
    if data_pos is None:
        data_pos = np.arange(num_const)
    if pic_idx is None:
        n_rows, n_cols, figsize = auto_constellation_map_param(num_const)
        pic_num = n_rows * n_cols
        pic_idx = np.linspace(start=0, stop=0 + num_const // (pic_num - 1) * (pic_num - 1), num=pic_num).astype(np.int32)
    else:
        n_rows, n_cols, figsize = auto_constellation_map_param(pic_idx.size)
        pic_num = n_rows * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, index in enumerate(pic_idx):
        ax = axes[i // n_cols, i % n_cols]
        draw_constellation_map(received=const[index], emit_pilot=const_ref[index],
                               ax=ax, title=f"constellation{data_pos[index]+1}", limit_border=2)
    fig.suptitle("data constellation")
    plt.tight_layout()
    plt.show()

# def plot_evm_vs_sub_carrier_idx(evm_avg:np.ndarray):
#     evm_mean_total = np.mean(evm_avg)
#     plt.figure(figsize=(10, 4))
#     plt.scatter(DATA_BINS, 20 * np.log10(evm_avg), marker='o', color='royalblue', s=1, alpha=0.5)
#     plt.axhline(20 * np.log10(evm_mean_total), linestyle='dotted', color='red')
#     plt.xlabel("Subcarrier Index")
#     plt.ylabel("EVM (dB)")
#     plt.title("EVM vs Subcarrier Index")
#     plt.grid(True)
#     plt.show()

# def pilot_quality_from_constellation(eq_const, ref):
#     """用中位数 SNR(dB) 作为该 pilot 的质量指标，返回线性质量 q（越大越好）。"""
#     evm_rms, _, snr_db = evm_and_snr(eq_const, ref)
#     snr_med_db = float(np.median(snr_db))
#     q = 10.0 ** (snr_med_db / 20.0)  # 线性质量
#     return q, snr_med_db
#
def plot_snr_over_time(snr_db_per_symbol: np.ndarray, title="SNR over OFDM symbols",*, pos: np.ndarray | None = None):
    if pos is None:
        pos = np.arange(snr_db_per_symbol.size)
    plt.figure(figsize=(10, 4))
    plt.plot(pos, snr_db_per_symbol, marker='o', linewidth=1.5)
    plt.xlabel("OFDM Symbol Index")
    plt.ylabel("SNR (dB)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_snr_over_subcarrier(snr_db_per_sc: np.ndarray, sc_idx, title="SNR over subcarriers"):
    plt.figure(figsize=(10, 4))
    plt.plot(sc_idx, snr_db_per_sc, linewidth=1.0)
    plt.xlabel("Subcarrier Index")
    plt.ylabel("SNR (dB)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_pre_post_ber(pre_ber: np.ndarray, post_ber: np.ndarray):
    """
    画出 pre_ber 和 post_ber 的变化情况
    pre_ber 和 post_ber 叠加在同一张图
    """
    indices = np.arange(len(pre_ber))
    plt.figure(figsize=(10, 4))
    plt.plot(indices, pre_ber, label="pre_ber", color="blue", alpha=0.5)
    plt.plot(indices, post_ber, label="post_ber", color="red", alpha=0.5)
    plt.xlabel("Index")
    plt.ylabel("BER")
    plt.title("pre_ber & post_ber vs index")
    plt.legend()
    plt.grid(True)
    plt.show()