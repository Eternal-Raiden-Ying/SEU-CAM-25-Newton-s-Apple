import os
from utils import record_signal,record_signal_with_error

fs = 48000

output_dir = r"D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\shakespare"
filename = "received_signal_shakeapace_block_pilot.npy"



if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("🎙 开始录音...")
    record_signal(t=32,filename=filename,pth=output_dir,fs=fs)
    print("录音完成")
