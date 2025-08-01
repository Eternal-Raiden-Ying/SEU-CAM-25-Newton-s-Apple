import os
from utils import record_signal

fs = 48000

output_dir = r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record"
filename = "ofdm_signal_chirp_l2_f24k_fs48k_N4096_S5"



if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("🎙 开始录音...")
    record_signal(t=5,filename=filename,pth=output_dir,fs=fs)
    print("录音完成")
