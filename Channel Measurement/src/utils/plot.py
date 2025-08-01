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
               ignore_zero: bool = True):

    if isinstance(freq, int) or isinstance(freq, float):
        if half:
            x = np.linspace(0, freq/2, signal.size)
        else:
            x = np.linspace(0, freq, signal.size)
    elif isinstance(freq, np.ndarray):
        x = freq
    else:
        raise TypeError(f"Unsupported type, expected num or np.ndarray, received {type(freq)}")

    if mode == 'Amplitude':
        mag = np.abs(fft(signal)).flatten()
        if ignore_zero:
            mask = np.where(mag > 0)
            mag = mag[mask]
            x = x[mask]
        y = 20 * np.log10(np.where(mag == 0, 1e-12, mag))
    elif mode == 'Phase':
        y = np.angle(fft(signal)).flatten()
    else:
        raise ValueError(f"valid mode in ['Amplitude', 'Phase'], got {mode} yet")

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


def draw_constellation_map(received, emit_pilot, mode='QPSK', *, pth=None, filename=None):
    """

    :param received: received signal in freq domain, without the conjugate part
    :param emit_pilot: emit pilot signal without the conjugate part
    :param pth: if the sig needs to save, specified
    :param filename:
    :return:
    """
    constellation_emit = emit_pilot
    constellations = received

    if mode == 'QPSK':
        red_mask = np.where(constellation_emit == 1 + 1j)
        green_mask = np.where(constellation_emit == -1 + 1j)
        blue_mask = np.where(constellation_emit == -1 - 1j)
        yellow_mask = np.where(constellation_emit == 1 - 1j)

        block_num = constellations.shape[0]
        fig, axes = plt.subplots(1, block_num, figsize=(5 * block_num, 5))
        for index, constellation in enumerate(constellations):
            real = np.real(constellation)
            imag = np.imag(constellation)
            groups = {
                'RED': {'real': real[red_mask], 'imag': imag[red_mask], 'color': 'red', 'label': '1+j'},
                'GREEN': {'real': real[green_mask], 'imag': imag[green_mask], 'color': 'green', 'label': '-1+j'},
                'BLUE': {'real': real[blue_mask], 'imag': imag[blue_mask], 'color': 'blue', 'label': '-1-j'},
                'YELLOW': {'real': real[yellow_mask], 'imag': imag[yellow_mask], 'color': 'yellow', 'label': '1-j'}
            }
            ax = axes[index]
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_title(f'constellation_{index + 1}')
            for k, data in groups.items():
                ax.scatter(data['real'], data['imag'], c=data['color'], label=data['label'], alpha=0.6, s=1)
            ax.grid(True)

        fig.tight_layout()
        if filename:
            plt.savefig(os.path.join(pth, filename), dpi=300) if pth else plt.savefig(filename, dpi=300)
        plt.show()

    else:
        raise ValueError("Unsupported mode, mode should be in ['QPSK',]")
