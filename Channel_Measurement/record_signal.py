import os
from module.utils import record_signal,record_signal_with_error

fs = 48000

output_dir = r"D:\Documents\Coding\Python\SEUCAM\Channel_Measurement\record\LDPC\temp"
filename = "test.npy"



if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input("按任意键开始录音")
    print("🎙 开始录音...")
    record_signal(t=75, filename=filename,pth=output_dir,fs=fs)
    print("录音完成")
