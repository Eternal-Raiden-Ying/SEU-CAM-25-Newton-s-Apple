import os
from module.utils import record_signal,record_signal_with_error

fs = 48000

output_dir = r"D:\Documents\Coding\Python\SEUCAM\Channel_Measurement\record\LDPC\temp"
filename = "[gyh-tzc]received_tiff_chirp_l2_10_24k_fs48k_N8192_cp1024_S8standard_R1-2_Z81_802.11n_A_no_scrambler_0.05-0.8_head_no_comb_1.npy"



if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("🎙 开始录音...")
    record_signal(t=49,filename=filename,pth=output_dir,fs=fs)
    print("录音完成")
