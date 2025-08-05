import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties

# 设置字体对象
ch_font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')   # 中文（Mac 示例）
en_font = FontProperties(family='DejaVu Sans')                               # 英文

def draw_in_TD(time, signal: np.ndarray,*,
               title: str = 'signal in time domain',
               ax: Axes = None,
               x_label: str = "",
               y_label: str = ""):

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
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label + ' (dB)') if 'dB' not in y_label and mode == 'Amplitude' else plt.ylabel(y_label)
        plt.grid()
        plt.show()
    else:
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label + ' (dB)') if 'dB' not in y_label and mode == 'Amplitude' else plt.ylabel(y_label)
        ax.grid(True)


def draw_constellation_map(received, emit_pilot, mode='QPSK',
                           title='constellation_map',*,
                           ax=None,pth=None, filename=None):
    """

    :param received: received signal in freq domain, without the conjugate part
    :param emit_pilot: emit pilot signal without the conjugate part
    :param pth: if the sig needs to save, specified
    :param filename:
    :return:
    """
    constellation_emit = emit_pilot.flatten()
    constellation = received.flatten()

    if mode == 'QPSK':
        red_mask = np.where(constellation_emit == 1 + 1j)
        green_mask = np.where(constellation_emit == -1 + 1j)
        blue_mask = np.where(constellation_emit == -1 - 1j)
        yellow_mask = np.where(constellation_emit == 1 - 1j)

        real = np.real(constellation)
        imag = np.imag(constellation)
        groups = {
            'RED': {'real': real[red_mask], 'imag': imag[red_mask], 'color': 'red', 'label': '1+j'},
            'GREEN': {'real': real[green_mask], 'imag': imag[green_mask], 'color': 'green', 'label': '-1+j'},
            'BLUE': {'real': real[blue_mask], 'imag': imag[blue_mask], 'color': 'blue', 'label': '-1-j'},
            'YELLOW': {'real': real[yellow_mask], 'imag': imag[yellow_mask], 'color': 'yellow', 'label': '1-j'}
        }
        if ax:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_title(title)
            for k, data in groups.items():
                ax.scatter(data['real'], data['imag'], c=data['color'], label=data['label'], alpha=0.6, s=1)
            ax.grid(True)
        else:
            plt.title(title)
            for k, data in groups.items():
                plt.scatter(data['real'], data['imag'], c=data['color'], label=data['label'], alpha=0.6, s=1)
            plt.grid()
            plt.xlim(-5,5)
            plt.ylim(-5,5)

        if filename:
            plt.savefig(os.path.join(pth, filename), dpi=300) if pth else plt.savefig(filename, dpi=300)

    else:
        raise ValueError("Unsupported mode, mode should be in ['QPSK',]")
