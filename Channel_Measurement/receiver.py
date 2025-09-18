import os
import argparse
import numpy as np
from module.receiver.receiver_stable import receiver
from module.receiver.receiver_develop import receiver as receiver_dev
from module.receiver.receiver_oop_dev import receiver as receiver_oop


project_dir = r"D:\Documents\Coding\Python\SEUCAM"
output_dir = os.path.join(project_dir, "Channel_Measurement/output/ldpc")
record_dir = os.path.join(project_dir, "Channel_Measurement/record")
data_dir = os.path.join(project_dir, "Channel_Measurement/data")
save_dir = os.path.join(project_dir, "Channel_Measurement/save")

if __name__ == "__main__":
    assert os.path.exists(project_dir), "specify your proj dir"
    dirs = [output_dir, record_dir, data_dir]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    rx_pth = os.path.join(record_dir, "LDPC", "interact",
                          "[tzc]received_tiff_chirp_l2_10_24k_fs48k_N8192_cp1024_S8diff_R1-2_Z27_802.11n_A_random256_front_0.8_head_1.npy")
    pilot_pth = os.path.join(save_dir, "pilot", "pilot_8different_N8192_fixed.npy")
    tx_file_path = os.path.join(data_dir, "answer.tiff")
    # rx_pth = r"D:\Documents\Coding\Python\SEUCAM\Channel_Measurement\test.npy"

    plot_opt = {
        'correlation':                      True,
        'impulse_response':                 True,
        'raw_pilot_constellation':          True,
        'corrected_pilot_constellation':    True,
        'data_constellation':               True,
        'unwrap':                           True,
        'received_signal':                  False,
        'BER_show':                         True,
        'snr_time_pilot':                   True,  # 导频阶段的平均 SNR(随符号)曲线
        'snr_time_comb':                    True,
        'snr_time_data':                    True,  # 数据阶段（判决导向统计）的 SNR(随符号)曲线
        'snr_over_sc':                      True,  # 跨子载波的平均 SNR 曲线 (data symbol)
        'freq_offset_interpolate':          True
    }

    print_opt = {
        'pilot_metric':                     True,
        'pilot_delta':                      False,
        'data_metric':                      True,
        'iter_verbose':                     True
    }

    suffix_map = {
        "tif": "tiff",
        "txt": "txt",
    }

    args = argparse.Namespace(
        # basic param
        fs=48000, N=8192, cp_len=1024,
        num_pilot=8, clockwise=False,
        chirp_len=2, chirp_l=10, chirp_h=24000,
        # file type
        head_bit=64, size_bit_w=40, type_bit_w=24, suffix_map={v: k for k, v in suffix_map.items()},
        # chirp param
        data_start=0, data_tail=819,
        # comb param
        use_comb=False, INTERVAL=None, COMB_PILOT_SEED_BASE=128,
        # pseudo pilot strategy
        edge_expand=32, max_pseudo_iter=20,
        # groundtruth settings
        groundtruth=True, tx_file_path=tx_file_path,
        # scrambler param
        use_scrambler=True, scrambler_seed=256, scrambler_mode='random', scrambler_bitwidth=None,
        # ldpc param
        ldpc_device="cuda", ldpc_batch=512,
        ldpc_standard="802.11n", ldpc_rate="1/2", ldpc_z=27, ldpc_ptype="A", ldpc_microbatch=256,
        ldpc_llr_clip=10.0, ldpc_max_iter=200,
        ldpc_verbose=False, ldpc_print_iter=True, ldpc_log_every=1, ldpc_check_every=1,
        # CPE PLL param
        pll_alpha=0.15, pll_beta=0.9,
        pll_alpha_min=0.05, pll_alpha_max=0.50,
        pll_snr_th_db=6.0, pll_snr_scale=4.0,
        pll_snr_th_min_db=3.0, pll_snr_mid_db=6.0, pll_snr_th_max_db=20.0,
        # sigma tracker
        sig_trk_per_sc=True, sig_trk_alpha_min=0.05, sig_trk_alpha_max=0.7, sig_trk_init_sigma=0.3,
        # frequency offset interpolate
        interp_mode='hold', interp_smooth=0.0,
        # plot settings
        plot=True, plot_opt=plot_opt,
        # print settings
        print_flag=True, print_opt=print_opt, print_len=64, print_pad='-'
    )
    rx = np.load(rx_pth).ravel()[args.fs:]
    if rx.dtype == np.int16:
        rx = rx.astype(np.float32)
        rx /= np.max(np.abs(rx))
    pilot = np.load(pilot_pth)
    decoded_info, info = receiver(rx, pilot, args)

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

