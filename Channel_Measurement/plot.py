import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

cpu_time = [61.8917, 178.2081, 530.9289]
gpu_time = [14.3213, 24.1026, 37.4051]
cpu_time.reverse()
gpu_time.reverse()
mpl.rcParams["font.family"] = "Times New Roman"

def plot_cpu_gpu_bars(
    cpu_time, gpu_time,
    group_labels=None,                 # 例如 ["Task A","Task B","Task C"]
    colors=("tab:blue", "tab:orange"), # CPU/GPU 颜色
    title=None,
    ylabel="Time (s)",
    ylim=None,
    annotate=True,                     # 是否在柱顶标注数值
    save_path=None, dpi=120
):
    cpu = np.asarray(cpu_time, dtype=float).ravel()
    gpu = np.asarray(gpu_time, dtype=float).ravel()
    if cpu.shape != (3,) or gpu.shape != (3,):
        raise ValueError("cpu_time 和 gpu_time 都应为 shape=(3,)。")

    n = 3
    x = np.arange(n)                 # 组的 x 位置：0,1,2
    width = 0.35                     # 每根柱的宽度

    fig, ax = plt.subplots(figsize=(6,4))
    # 两组柱：左右各偏移半个宽度
    bars_cpu = ax.bar(x - width/2, cpu, width, label="CPU", color=colors[0])
    bars_gpu = ax.bar(x + width/2, gpu, width, label="GPU", color=colors[1])

    # 组标签与坐标
    if group_labels is None:
        group_labels = [f"Group {i+1}" for i in range(n)]
    ax.set_xticks(x, group_labels, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title, fontsize=18)
    if ylim:
        ax.set_ylim(*ylim)

    # 右上角图例
    ax.legend(loc="upper right", fontsize=14)

    # 可选：柱顶标注
    if annotate:
        def _annotate(bars):
            for b in bars:
                h = b.get_height()
                ax.annotate(f"{h:.3g}",
                            xy=(b.get_x() + b.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=12,fontweight='bold')
        _annotate(bars_cpu)
        _annotate(bars_gpu)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi)
    return fig, ax


if __name__ == "__main__":
    plot_cpu_gpu_bars(
        cpu_time, gpu_time,
        group_labels=["low-SNR", "mid-SNR", "high-SNR"],
        title="CPU vs GPU Decoding Time",
        ylabel="Time (s)",
        annotate=True,
        ylim=(0,650),
        colors=(
            (168/256,201/256,220/256),
            (251/256,118/256,119/256)
        )
    )
    plt.show()
