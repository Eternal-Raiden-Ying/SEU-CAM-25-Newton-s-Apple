import os
import argparse
import numpy as np
from module.receiver.receiver_scrambler_ldpc_comb_pilot_part_valid import receiver
from module.receiver.receiver_develop import receiver as receiver_dev
from module.receiver.receiver_oop_dev import receiver as receiver_oop


project_dir = r"D:\Documents\Coding\Python\SEUCAM"
output_dir = os.path.join(project_dir, "Channel_Measurement/output/ldpc")
record_dir = os.path.join(project_dir, "Channel_Measurement/record")
data_dir = os.path.join(project_dir, "Channel_Measurement/data")

if __name__ == "__main__":
    assert os.path.exists(project_dir), "specify your proj dir"
    dirs = [output_dir, record_dir, data_dir]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    rx_pth = r"D:\Documents\Coding\Python\SEUCAM\Channel_Measurement\record\temp\[tzc]received_tiff_chirp_l2_10_24k_fs48k_N8192_cp1024_S8diff_R1-2_Z27_802.11n_A_random256_front_0.8_head_1.npy"
    pilot_pth = r'D:\Documents\Coding\Python\SEUCAM\Channel Measurement\record\LDPC\pilot_different_txt820_seed256_part0.8_comb.npy'

    plot_opt = {
        'correlation':                      False,
        'impulse_response':                 False,
        'raw_pilot_constellation':          False,
        'corrected_pilot_constellation':    False,
        'data_constellation':               True,
        'unwrap':                           True,
        'received_signal':                  True,
        'BER_show':                         True,
        'snr_time_pilot':                   False,  # 导频阶段的平均 SNR(随符号)曲线
        'snr_time_comb':                    False,
        'snr_time_data':                    False,  # 数据阶段（判决导向统计）的 SNR(随符号)曲线
        'snr_over_sc':                      False,  # 跨子载波的平均 SNR 曲线 (data symbol)
    }

    args = argparse.Namespace(
        # basic param
        fs=48000, N=8192, cp_len=1024, num_pilot=8,
        chirp_len=2, chirp_l=10, chirp_h=24000,
        # chirp param
        data_start=0, data_tail=819,
        # comb param
        INTERVAL=10, COMB_PILOT_SEED_BASE=128,
        # groundtruth
        groundtruth=True, head_bit=0,
        tx_file_path=r"D:\Documents\Coding\Python\SEUCAM\Channel_Measurement\data\answer.tiff",
        # scrambler param
        scrambler_seed=256, scrambler_mode='random', scrambler_bitwidth=None, clockwise=False,
        # ldpc param
        ldpc_standard="802.11n", ldpc_rate="1/2", ldpc_z=27, ldpc_ptype="A",
        ldpc_device="cuda", ldpc_llr_clip=10.0, ldpc_max_iter=600,
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
        data_seg_len=8,
        # plot settings
        plot=False, plot_opt=plot_opt,
        # print settings
        print_flag=True, print_len=64, print_pad='-'
    )
    rx = np.load(rx_pth)
    pilot = np.load(pilot_pth)
    decoded_info, info = receiver_dev(rx, pilot, args)
    print(f"ldpc iter: {info['ldpc_iter']}")
    if args.groundtruth:
        print(f"pre_ber: {info['pre_ber']}")
        print(f"post_ber: {info['post_ber']}")

    bytes = np.packbits(decoded_info.flatten())
    with open(output_dir + "/unknown1.tiff", 'wb') as file:
        file.write(bytes.tobytes())

