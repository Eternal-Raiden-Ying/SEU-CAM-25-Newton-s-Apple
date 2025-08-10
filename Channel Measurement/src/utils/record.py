import os
import numpy as np
import sounddevice as sd

default_fs = 48000


def record_signal(t, filename, pth, fs=default_fs,*, channel=1,dtype='float64'):
    """
        record a signal and save as .npy file
    :param t: record duration
    :param filename:
    :param pth: directory to save the file
    :param fs: sampling freq
    :param channel: record channel
    :param dtype: default fp64
    :return: None
    """
    if not os.path.exists(pth):
        os.makedirs(pth)
    rx = sd.rec(t*fs, samplerate=fs, channels=channel, dtype=dtype)
    sd.wait()
    rx = rx.flatten()

    np.save(os.path.join(pth,filename),rx)


def record_signal_with_error(t, actual_fs, filename, pth, fs=default_fs,*, channel=1,dtype='float64'):
    """
        make an artificial fs offset and then record,
        get you same quantity of data but at a different fs
    :param actual_fs:  actual sampling freq
    :param t: record duration
    :param filename:
    :param pth: directory to save the file
    :param fs: influence the length of data you record
    :param channel: record channel
    :param dtype: default fp64
    :return: None
    """
    if not os.path.exists(pth):
        os.makedirs(pth)
    rx = sd.rec(t*fs, samplerate=actual_fs, channels=channel, dtype=dtype)
    sd.wait()
    rx = rx.flatten()

    np.save(os.path.join(pth,filename),rx)