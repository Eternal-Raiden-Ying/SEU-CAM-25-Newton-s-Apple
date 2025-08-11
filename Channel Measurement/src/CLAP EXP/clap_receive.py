import numpy as np
import sounddevice as sd
import os
import time

output_dir = r"/Channel Measurement/record"
filename = "clap_once_bedroom"

fs = 48000      # sampling rate
duration = 3    # time(s)

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    time.sleep(1)
    print("recording ...")
    rx = sd.rec(int(fs * duration), samplerate=fs, channels=1)
    sd.wait()
    print("done")
    np.save(os.path.join(output_dir, f"{filename}.npy"), rx)