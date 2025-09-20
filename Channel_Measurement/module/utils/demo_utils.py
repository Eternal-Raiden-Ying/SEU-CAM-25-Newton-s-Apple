import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # MP4 可用 FFMpegWriter（需装 ffmpeg）

mpl.rcParams["font.family"] = "Times New Roman"

# 可选：GIF 事后量化压缩（显著减小体积）
def _quantize_gif(path_in, path_out=None, colors=128, optimize=True, duration_ms=None):
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        return  # 没装 Pillow 就跳过
    im = Image.open(path_in)
    frames = []
    for fr in ImageSequence.Iterator(im):
        # 转成自适应调色板，最多 colors 色（GIF 上限 256）
        q = fr.convert("P", palette=Image.ADAPTIVE, colors=int(colors))
        frames.append(q)
    if not frames:
        return
    save_to = path_out or path_in
    duration = duration_ms if duration_ms is not None else im.info.get("duration", 40)
    frames[0].save(
        save_to, save_all=True, append_images=frames[1:], loop=0,
        optimize=optimize, duration=duration
    )

def animate_scatter_panels_classes(
    frames,                                  # frames[t][p][c] = (x, y)
    panel_titles=("P1","P2","P3","P4"),
    class_labels=("Type 1","Type 2","Type 3","Type 4"),
    colors=("C0","C1","C2","C3"),
    markers=("o","o","o","o"),
    s=20, alpha=0.9,
    xlim=None, ylim=None,                    # (min,max) 或长度为4
    interval=40, fps=25,
    save_path=None, suptitle=None,
    xticks=None, yticks=None,
    n_xticks=None, n_yticks=None,
    add_axes="lines",                        # None / "lines" / "spines"
    arrows=False,

    # —— 新增：动态字幕（动图下方关键词/短语） ——
    captions=None,                           # None / 可迭代(长度>=1) / 可调用 f(t)->str
    caption_y=0.02,                          # 相对图高位置（0=底部，1=顶部）
    caption_kw=None,                         # dict，例如 {"fontsize":12,"fontweight":"bold"}

    # —— 新增：体积控制 ——
    fig_size=(16, 4),                        # 减小尺寸可显著降体积
    dpi=80,                                 # 保存时 DPI，适当降低（如 80）
    every_n=1,                               # 帧抽样：每隔 n 帧取一帧（例如 2 可减半）
    gif_colors=128,                          # GIF 量化颜色数（<=256，越小越省）
    gif_optimize=True,                       # GIF 保存后再用 Pillow 优化
):
    # —— 预处理帧（抽帧） ——
    frames = list(frames)
    if every_n > 1:
        frames = frames[::int(every_n)]
    if not frames:
        raise ValueError("frames 不能为空。")
    T = len(frames)

    # 结构校验
    for t, ft in enumerate(frames):
        if len(ft) != 4:
            raise ValueError(f"第 {t} 帧应包含 4 个子图数据。")
        for p, fp in enumerate(ft):
            if len(fp) != 4:
                raise ValueError(f"第 {t} 帧的第 {p} 个子图应包含 4 个类别数据。")
            for c, pair in enumerate(fp):
                if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                    raise ValueError(f"frames[{t}][{p}][{c}] 必须是 (x, y)。")

    if len(panel_titles) != 4 or len(class_labels) != 4 or len(colors) != 4 or len(markers) != 4:
        raise ValueError("panel_titles/class_labels/colors/markers 都必须各 4 个。")

    # 归一化 x/y 范围
    def _norm_lim(lim):
        if lim is None: return None
        if isinstance(lim, (tuple, list)) and len(lim) == 2 and np.isscalar(lim[0]) and np.isscalar(lim[1]):
            return [tuple(lim)]*4
        if isinstance(lim, (tuple, list)) and len(lim) == 4:
            return [tuple(v) if v is not None else None for v in lim]
        raise ValueError("xlim/ylim 必须是 (min,max) 或 长度为4 的列表。")

    xlims = _norm_lim(xlim)
    ylims = _norm_lim(ylim)

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

    def _linspace_ticks(lim, n):
        if n is None: return None
        n = max(int(n), 2)
        return np.linspace(lim[0], lim[1], n)

    # —— 画布/子图；为字幕预留下边距 ——
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    # fig.patch.set_alpha(0)
    fig.subplots_adjust(bottom=0.16)
    if suptitle:
        fig.suptitle(suptitle, y=0.98, fontsize=24, fontweight="bold")

    # —— 字幕 Text Artist（随帧更新） ——
    footer_ax = fig.add_axes([0.08, 0.0, 0.84, 0.12])  # [left,bottom,width,height] in figure coords
    footer_ax.axis("off")

    # 页脚中的字幕 Text（在页脚轴的坐标系中居中）
    caption_obj = None
    if captions is not None:
        cap_style = dict(fontsize=12, ha="center", va="center")
        if isinstance(caption_kw, dict):
            cap_style.update(caption_kw)
        caption_obj = footer_ax.text(0.5, 0.5, "", **cap_style)

        # 小工具：取某帧字幕
        def _get_caption(t):
            if callable(captions):
                return str(captions(t))
            captions_seq = list(captions)
            return str(captions_seq[t % len(captions_seq)])

    scatts = []  # [[sc_p0_c0..c3], ...]
    for p in range(4):
        ax = axes[p//2, p%2]
        # ax.set_facecolor('none')
        ax.set_title(panel_titles[p])
        ax.set_xlim(*xlims[p]); ax.set_ylim(*ylims[p])
        if p == 2 or p == 3: ax.set_xlabel("real", fontsize=14)
        if p == 0 or p == 2: ax.set_ylabel("imag", fontsize=14)

        # 统一刻度（仅初始化一次）
        if xticks is not None:
            ax.set_xticks(xticks)
        elif n_xticks is not None:
            ax.set_xticks(_linspace_ticks(xlims[p], n_xticks))
        if yticks is not None:
            ax.set_yticks(yticks)
        elif n_yticks is not None:
            ax.set_yticks(_linspace_ticks(ylims[p], n_yticks))

        # 0 轴
        if add_axes == "lines":
            ax.axhline(0, linewidth=1)
            ax.axvline(0, linewidth=1)
            if arrows:
                ax.annotate('', xy=(xlims[p][1], 0), xytext=(xlims[p][0], 0),
                            arrowprops=dict(arrowstyle='->', linewidth=1))
                ax.annotate('', xy=(0, ylims[p][1]), xytext=(0, ylims[p][0]),
                            arrowprops=dict(arrowstyle='->', linewidth=1))
        elif add_axes == "spines":
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        row = []
        for c in range(4):
            sc = ax.scatter([], [], s=s, c=colors[c], marker=markers[c], alpha=alpha,
                            label=class_labels[c] if p == 3 else None)
            row.append(sc)
        scatts.append(row)

    axes[1,1].legend(loc="lower right", fontsize=14)

    def init():
        for row in scatts:
            for sc in row:
                sc.set_offsets(np.empty((0, 2)))
        if caption_obj is not None:
            caption_obj.set_text("")
        return tuple(sc for row in scatts for sc in row) + ((caption_obj,) if caption_obj else ())

    def update(t):
        for p in range(4):
            for c in range(4):
                x, y = frames[t][p][c]
                x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
                ofs = np.empty((0, 2)) if x.size == 0 else np.column_stack([x, y])
                scatts[p][c].set_offsets(ofs)
        if caption_obj is not None:
            caption_obj.set_text(_get_caption(t))
        return tuple(sc for row in scatts for sc in row) + ((caption_obj,) if caption_obj else ())

    anim = FuncAnimation(fig, update, init_func=init, frames=T, interval=interval, blit=True)

    if save_path:
        if save_path.lower().endswith(".gif"):
            writer = PillowWriter(fps=fps)
            # 注意：dpi 会显著影响体积
            anim.save(save_path, writer=writer, dpi=dpi)
            # 事后量化压缩（显著减小体积）
            if gif_colors is not None or gif_optimize:
                _quantize_gif(save_path, colors=gif_colors or 256, optimize=gif_optimize,
                              duration_ms=int(1000/fps))
        elif save_path.lower().endswith(".mp4"):
            from matplotlib.animation import FFMpegWriter   # 需系统安装 ffmpeg
            writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=None)  # H.264 通常更小更清晰
            anim.save(save_path, writer=writer, dpi=dpi)
        else:
            raise ValueError("仅支持 .gif 或 .mp4")

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