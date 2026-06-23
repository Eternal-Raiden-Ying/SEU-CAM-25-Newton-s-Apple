import os
import numpy as np
import soundfile as sf

from module.receiver.receiver_stable import receiver
from module.receiver.receiver_oop_dev import receiver as receiver_oop
from module.cfg.config import ReceiverConfig, OFDMConfig, ChirpConfig, HeaderConfig, ScramblerConfig, LDPCConfig

project_dir = r"D:\Documents\Coding\Python\SEUCAM"
output_dir = os.path.join(project_dir, "Channel_Measurement/output/ldpc")
assets_dir = os.path.join(project_dir, "Channel_Measurement/assets")

if __name__ == "__main__":
    assert os.path.exists(project_dir), "specify your proj dir"
    os.makedirs(output_dir, exist_ok=True)

    rx_pth = os.path.join(project_dir, "Channel_Measurement/record", "LDPC", "temp", "[zrh]rx.wav")
    pilot_pth = os.path.join(assets_dir, "pilots", "pilot_STANDARD_freq_domain.npy")

    # Build ReceiverConfig with production defaults
    cfg = ReceiverConfig(
        ofdm=OFDMConfig(),
        chirp=ChirpConfig(),
        header=HeaderConfig(),
        scrambler=ScramblerConfig(),
        ldpc=LDPCConfig(print_iter=True),
    )

    # ---- Load recording ----
    if rx_pth.endswith(".wav"):
        rx_raw, sr = sf.read(rx_pth)
        rx = rx_raw[:, 0] if rx_raw.ndim == 2 else rx_raw
        rx = rx.astype(np.float64)
        print("Loaded recording:", rx_pth)
        print("Sample rate fs =", sr)
    elif rx_pth.endswith(".npy"):
        rx = np.load(rx_pth).ravel()
        if rx.dtype == np.int16:
            rx /= np.max(np.abs(rx))
        print("Loaded npy file:", rx_pth)
    else:
        raise ValueError(f"Unsupported format: {rx_pth}")

    pilot = np.load(pilot_pth)
    decoded_info, info = receiver(rx, pilot, cfg)

    print(f"ldpc iter: {info['ldpc_iter']}")
    if cfg.groundtruth:
        print(f"post_ber: {info['post_ber']}")

    out_bytes = np.packbits(decoded_info.flatten())
    if cfg.type_bit_w:
        suffix_str = cfg.suffix_map[info['type_suffix']]
        output_filename = f'unknown.{suffix_str}'
    else:
        output_filename = 'unknown.tiff'

    with open(os.path.join(output_dir, output_filename), 'wb') as file:
        file.write(out_bytes.tobytes())
