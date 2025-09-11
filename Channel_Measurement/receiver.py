import os
import argparse
import soundfile as sf
import numpy as np
from module.receiver.receiver_scrambler_ldpc_comb_pilot_part_valid import receiver
from module.receiver.receiver_develop import receiver as receiver_dev
from module.receiver.receiver_oop_dev import receiver as receiver_oop


project_dir = r"D:\Pycharm\SEU-CAM-25-Newton-s-Apple"
output_dir = os.path.join(project_dir, "Channel_Measurement/output/ldpc")
record_dir = os.path.join(project_dir, "Channel_Measurement/record")
data_dir = os.path.join(project_dir, "Channel_Measurement/data")

if __name__ == "__main__":
    assert os.path.exists(project_dir), "specify your proj dir"
    dirs = [output_dir, record_dir, data_dir]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    rx_pth = os.path.join("record", "LDPC",
                          "[wmh-smy]received_tiff_chirp_l2_10_24k_fs48k_N8192_cp1024_S8same_R1-2_Z81_802.11n_A_no_scrambler_0.05-0.8_head_no_comb_4.npy")
    pilot_pth = os.path.join("save", "pilot", "wmh-pilot_STANDARD_freq_domain.npy")
    tx_file_path = os.path.join("data", "answer.tiff")

    plot_opt = {
        'correlation':                      True,
        'impulse_response':                 True,
        'raw_pilot_constellation':          True,
        'corrected_pilot_constellation':    True,
        'data_constellation':               True,
        'unwrap':                           True,
        'received_signal':                  True,
        'BER_show':                         True,
        'snr_time_pilot':                   True,  # 导频阶段的平均 SNR(随符号)曲线
        'snr_time_comb':                    True,
        'snr_time_data':                    True,  # 数据阶段（判决导向统计）的 SNR(随符号)曲线
        'snr_over_sc':                      True,  # 跨子载波的平均 SNR 曲线 (data symbol)
    }

    suffix_map = {
        "tif": "tiff",
        "txt": "txt",
    }


    args = argparse.Namespace(
        # basic param
        fs=48000, N=8192, cp_len=1024, num_pilot=8,
        chirp_len=2, chirp_l=10, chirp_h=24000,
        # chirp param
        data_start=204, data_tail=819,
        # comb param
        INTERVAL=None, COMB_PILOT_SEED_BASE=128, use_comb=False,
        # groundtruth
        groundtruth=True, head_bit=64, size_bit_w=40, type_bit_w=24, suffix_map={v: k for k,v in suffix_map.items()},
        tx_file_path=tx_file_path,
        # scrambler param
        use_scrambler=False, scrambler_seed=256, scrambler_mode='random', scrambler_bitwidth=None, clockwise=False,
        # ldpc param
        ldpc_standard="802.11n", ldpc_rate="1/2", ldpc_z=81, ldpc_ptype="A",
        ldpc_device="cuda", ldpc_llr_clip=10.0, ldpc_max_iter=200,
        ldpc_verbose=False, ldpc_log_every=1, ldpc_check_every=1,
        ldpc_microbatch=256, ldpc_batch=512, ldpc_print_iter=True,
        # CPE PLL param
        pll_alpha=0.15, pll_snr_th_db=6.0,
        pll_alpha_min=0.05, pll_alpha_max=0.50,
        pll_snr_th_min_db=3.0, pll_snr_th_max_db=20.0,
        pll_beta=0.9, pll_snr_mid_db=6.0, pll_snr_scale=4.0,
        # sigma tracker
        sig_trk_per_sc=True, sig_trk_alpha_min=0.05, sig_trk_alpha_max=0.7, sig_trk_init_sigma=0.3,
        # fake pilot
        edge_expand=32, max_pseudo_iter=20,
        # plot settings
        plot=True, plot_opt=plot_opt,
        # print settings
        print_flag=True, print_len=64, print_pad='-', iter_verbose=True
    )

    # ---- 加载录音文件 ----
    if rx_pth.endswith(".wav"):
        # wav 文件读取
        rx_raw, sr = sf.read(rx_pth)  # sr: 采样率
        # 如果是立体声，取第1通道；否则直接使用
        rx = rx_raw[:, 0] if rx_raw.ndim == 2 else rx_raw
        rx = rx.astype(np.float64)
        print("已加载录音文件：", rx_pth)
        print("采样率 fs =", sr)
    elif rx_pth.endswith(".npy"):
        # npy 文件读取
        rx = np.load(rx_pth)  # 原本逻辑保留
        print("已加载npy文件：", rx_pth)
    else:
        raise ValueError(f"不支持的文件格式: {rx_pth}")

    pilot = np.load(pilot_pth)
    decoded_info, info = receiver_dev(rx, pilot, args)
    print(f"ldpc iter: {info['ldpc_iter']}")
    if args.groundtruth:
        print(f"post_ber: {info['post_ber']}")

    bytes = np.packbits(decoded_info.flatten())
    if getattr(args, 'type_bit_w', 0):
        suffix_str = suffix_map[info['type_suffix']]
        output_filename = f'unknown.{suffix_str}'
    else:
        output_filename = 'unknown.tiff'

    # 写入文件
    with open(os.path.join(output_dir, output_filename), 'wb') as file:
        file.write(bytes.tobytes())

