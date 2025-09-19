import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # MP4 可用 FFMpegWriter（需装 ffmpeg）

def animate_scatter_panels_classes(
    frames,  # 结构：frames[t][p][c] = (x, y)；t:帧，p:面板0..3，c:类别0..3
    panel_titles=("P1","P2","P3","P4"),
    class_labels=("Type 1","Type 2","Type 3","Type 4"),
    colors=("C0","C1","C2","C3"),
    markers=("o","o","o","o"),
    s=20, alpha=0.9,
    xlim=None, ylim=None,            # (min,max) 或长度为4的列表
    interval=40, fps=25,
    save_path=None, suptitle=None,
    # —— 新增：刻度与轴线控制 ——
    xticks=None, yticks=None,        # 刻度位置数组/列表（优先级最高）
    n_xticks=None, n_yticks=None,    # 刻度个数（含端点；若上面未给，才会生效）
    add_axes="lines",                # None / "lines" / "spines"
    arrows=False                     # 是否给 0 轴加箭头（建议与 "lines" 搭配）
):
    frames = list(frames)
    if not frames:
        raise ValueError("frames 不能为空。")
    T = len(frames)
    for t, ft in enumerate(frames):
        if len(ft) != 4:
            raise ValueError(f"第 {t} 帧应包含 4 个子图数据。")
        for p, fp in enumerate(ft):
            if len(fp) != 4:
                raise ValueError(f"第 {t} 帧的第 {p} 个子图应包含 4 个类别数据。")
            for c, pair in enumerate(fp):
                if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                    raise ValueError(f"frames[{t}][{p}][{c}] 必须是 (x, y)。")

    if len(panel_titles) != 4:  raise ValueError("panel_titles 必须 4 个。")
    if len(class_labels) != 4:  raise ValueError("class_labels 必须 4 个。")
    if len(colors) != 4:        raise ValueError("colors 必须 4 个。")
    if len(markers) != 4:       raise ValueError("markers 必须 4 个。")

    # 归一化 x/y 范围
    def _norm_lim(lim, name):
        if lim is None: return None
        if isinstance(lim, (tuple, list)) and len(lim) == 2 and np.isscalar(lim[0]) and np.isscalar(lim[1]):
            return [tuple(lim)] * 4
        if isinstance(lim, (tuple, list)) and len(lim) == 4:
            return [tuple(v) if v is not None else None for v in lim]
        raise ValueError(f"{name} 必须是 (min,max) 或 长度为4 的列表。")

    xlims = _norm_lim(xlim, "xlim")
    ylims = _norm_lim(ylim, "ylim")

    # 自动按“每个面板聚合其 4 类、全部帧”的数据计算范围（仅在未提供时）
    def _auto_limits_for_panel(p, is_x=True):
        vals = []
        for t in range(T):
            for c in range(4):
                arr = np.asarray(frames[t][p][c][0 if is_x else 1]).ravel()
                if arr.size: vals.append(arr)
        if not vals: return (-1, 1)
        v = np.concatenate(vals)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        pad = (vmax - vmin) * 0.05 + 1e-9
        return (vmin - pad, vmax + pad)

    if xlims is None or ylims is None:
        if xlims is None: xlims = [None]*4
        if ylims is None: ylims = [None]*4
        for p in range(4):
            if xlims[p] is None: xlims[p] = _auto_limits_for_panel(p, True)
            if ylims[p] is None: ylims[p] = _auto_limits_for_panel(p, False)

    # 工具：根据 lim 和 n_ticks 生成均匀刻度
    def _linspace_ticks(lim, n):
        if n is None: return None
        n = int(n)
        n = max(n, 2)  # 至少含端点2个
        return np.linspace(lim[0], lim[1], n)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    if suptitle: fig.suptitle(suptitle, y=0.98)

    scatts = []  # [[sc_p0_c0..c3], ...]
    for p, ax in enumerate(axes):
        ax.set_title(panel_titles[p])
        ax.set_xlim(*xlims[p]); ax.set_ylim(*ylims[p])
        ax.set_xlabel("real")
        if p == 0: ax.set_ylabel("imag")

        # —— 统一刻度（仅初始化设置一次，动画期间不再改动） ——
        # 优先使用显式 xticks/yticks，其次使用 n_xticks/n_yticks 均匀取点
        if xticks is not None:
            ax.set_xticks(xticks)
        elif n_xticks is not None:
            ax.set_xticks(_linspace_ticks(xlims[p], n_xticks))
        if yticks is not None:
            ax.set_yticks(yticks)
        elif n_yticks is not None:
            ax.set_yticks(_linspace_ticks(ylims[p], n_yticks))

        # —— 加 xy 轴（只做一次） ——
        if add_axes == "lines":
            # 过原点的横纵轴
            ax.axhline(0, linewidth=1)
            ax.axvline(0, linewidth=1)
            if arrows:
                # 简易箭头（确保范围包含 0 才可见）
                ax.annotate('', xy=(xlims[p][1], 0), xytext=(xlims[p][0], 0),
                            arrowprops=dict(arrowstyle='->', linewidth=1))
                ax.annotate('', xy=(0, ylims[p][1]), xytext=(0, ylims[p][0]),
                            arrowprops=dict(arrowstyle='->', linewidth=1))
        elif add_axes == "spines":
            # 脊柱穿过零点（范围需包含0）
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        # add_axes 为 None 时不做额外处理

        row = []
        for c in range(4):
            sc = ax.scatter([], [], s=s, c=colors[c], marker=markers[c], alpha=alpha,
                            label=class_labels[c] if p == 0 else None)
            row.append(sc)
        scatts.append(row)

    # 只在第一个面板放图例，避免重复
    axes[3].legend(loc="lower right")

    def init():
        for row in scatts:
            for sc in row:
                sc.set_offsets(np.empty((0, 2)))
        return tuple(sc for row in scatts for sc in row)

    def update(t):
        for p in range(4):
            for c in range(4):
                x, y = frames[t][p][c]
                x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
                ofs = np.empty((0, 2)) if x.size == 0 else np.column_stack([x, y])
                scatts[p][c].set_offsets(ofs)
        return tuple(sc for row in scatts for sc in row)

    anim = FuncAnimation(fig, update, init_func=init, frames=T, interval=interval, blit=True)

    if save_path:
        if save_path.lower().endswith(".gif"):
            writer = PillowWriter(fps=fps)
        elif save_path.lower().endswith(".mp4"):
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps)
        else:
            raise ValueError("仅支持 .gif 或 .mp4")
        anim.save(save_path, writer=writer)

    return anim



def complex_normal(
    shape,
    mean=0.0,              # 标量或 (mean_real, mean_imag)；也可传可广播到 shape 的数组
    var=1.0,               # 标量或 (var_real, var_imag)；方差需 >= 0
    dtype=np.complex128,   # np.complex64 或 np.complex128
    rng=None               # None / int(随机种子) / np.random.Generator
):
    """
    生成复数数组 Z = X + iY
    其中 X ~ N(mean_real, var_real), Y ~ N(mean_imag, var_imag)

    参数:
      - shape: 元组或整数，输出形状
      - mean: 标量或长度为2的元组/列表/数组 (mean_real, mean_imag)
              也可传入可广播到 shape 的数组
      - var:  标量或长度为2的元组/列表/数组 (var_real, var_imag)
              也可传入可广播到 shape 的数组；必须非负
      - dtype: np.complex64 或 np.complex128
      - rng:  None / int / np.random.Generator
    """
    # 处理 RNG
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)  # rng 为 None 或 int 都可以

    # 拆分均值与方差（允许标量或二元组）
    def split2(x):
        if np.isscalar(x):
            return x, x
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2:
            return x[0], x[1]
        raise ValueError("mean/var 应为标量或长度为 2 的 (real, imag)。")

    mean_r, mean_i = split2(mean)
    var_r,  var_i  = split2(var)

    # 校验方差
    if np.any(np.asarray(var_r) < 0) or np.any(np.asarray(var_i) < 0):
        raise ValueError("方差 var 必须 >= 0。")

    # 生成标准正态并仿射变换（支持广播）
    std_r = np.sqrt(var_r)
    std_i = np.sqrt(var_i)

    real = gen.standard_normal(size=shape) * std_r + mean_r
    imag = gen.standard_normal(size=shape) * std_i + mean_i

    # 按 dtype 输出（float32+float32j 或 float64+float64j）
    if dtype == np.complex64:
        real = np.asarray(real, dtype=np.float32)
        imag = np.asarray(imag, dtype=np.float32)
    elif dtype == np.complex128:
        real = np.asarray(real, dtype=np.float64)
        imag = np.asarray(imag, dtype=np.float64)
    else:
        raise ValueError("仅支持 dtype 为 np.complex64 或 np.complex128。")

    return real + 1j * imag