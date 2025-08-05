import os
import numpy as np
import sounddevice as sd
import logging

project_record_dir = r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record"
default_fs = 48000
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename=r'D:\Documents\Coding\Python\SEUCAM\Channel Measurement\logging\record.log',
                    filemode='w')


def record_signal(t, filename, pth=project_record_dir, fs=default_fs,*, channel=1,dtype='float64'):
    if not os.path.exists(pth):
        os.makedirs(pth)
    logging.info(f"record signal for {t}s, result in f{os.path.join(pth,filename)}")
    rx = sd.rec(t*fs, samplerate=fs, channels=1, dtype=dtype)
    sd.wait()
    rx = rx.flatten()

    np.save(os.path.join(pth,filename),rx)


def record_signal_with_error(t, actual_fs, filename, pth=project_record_dir, fs=default_fs,*, channel=1,dtype='float64'):
    if not os.path.exists(pth):
        os.makedirs(pth)
    logging.info(f"record signal for {t}s, result in f{os.path.join(pth,filename)}")
    rx = sd.rec(t*fs, samplerate=actual_fs, channels=1, dtype=dtype)
    sd.wait()
    rx = rx.flatten()

    np.save(os.path.join(pth,filename),rx)