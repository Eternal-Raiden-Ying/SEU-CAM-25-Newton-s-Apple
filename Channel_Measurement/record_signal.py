import os
from module.utils import record_signal,record_signal_with_error

fs = 48000

output_dir = r"D:\Documents\Coding\Python\SEUCAM\Channel_Measurement\record\temp"
filename = "[tzc]received_tiff_chirp_l2_10_24k_fs48k_N8192_cp1024_S8same_R1-2_Z81_802.11n_A_random256_front_0.8_head_1.npy"



if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("🎙 开始录音...")
    record_signal(t=49,filename=filename,pth=output_dir,fs=fs)
    print("录音完成")
