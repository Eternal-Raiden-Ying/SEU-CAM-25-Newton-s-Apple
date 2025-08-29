import os
from utils import record_signal,record_signal_with_error

fs = 48000

output_dir = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple\Channel Measurement\record\LDPC"
filename = "received_txt_chirp_l2_10_24k_fs48k_N8192_cp1024_S8diff_R3-4_Z27_802.11n_A_random_middle_0.8_head_3.npy"

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("🎙 开始录音...")
    record_signal(t= 34 ,filename=filename,pth=output_dir,fs=fs)
    print("录音完成")
