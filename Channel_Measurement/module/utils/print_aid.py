import numpy as np
from typing import Any, Dict, Optional, Sequence, Union
from collections.abc import Sequence as _Sequence

def print_padded(text: str, length: int, pad_char: str = ' ', truncate: bool = False) -> None:
    """
    将 text 两边各加两个空格，然后用 pad_char 填充到指定总长度后打印。
    - text: 要打印的字符串（不会为 None）
    - length: 目标总长度（包含两侧的两个空格）
    - pad_char: 用于填充的字符串（可以是多个字符，会被重复并切片精确填充）
    - truncate:
        * False（默认）：若内容超长则直接打印完整内容（不截断、不填充）
        * True：若内容超长则使用省略号 '...' 代替尾部（尽量保留左右两个空格）
    说明/边界处理：
      - 当 length <= 0 时直接打印带两空格的原文。
      - 当 length 足够时，左右填充尽量均匀（右侧在总填充为奇数时多一个字符）。
      - 若 length 太小以致无法同时保留两空格与完整 '...'，会回退为尽可能的短形式（优先保证能显示一些可读信息）。
    """
    if length <= 0:
        print(f"  {text}  ")
        return

    left_right_spaces = 4  # 两侧各两个空格
    ell = "..."

    content = f"  {text}  "
    content_len = len(content)

    def make_pad(n: int) -> str:
        if n <= 0:
            return ''
        if not pad_char:
            return ' ' * n
        reps = (n // len(pad_char)) + 2
        return (pad_char * reps)[:n]

    # 内容比期望短或刚好：正常填充
    if content_len <= length:
        total_pad = length - content_len
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        left = make_pad(left_pad)
        right = make_pad(right_pad)
        print(f"{left}{content}{right}")
        return

    # content_len > length 情况
    if not truncate:
        # 不截断，则直接打印完整内容（含两侧空格）
        print(content)
        return

    # 使用省略号来尽量适配长度（保留两侧空格优先）
    # 可用于 text 的字符数 = length - (两空格) - len(ellipsis)
    avail_for_text = length - left_right_spaces - len(ell)

    if avail_for_text > 0:
        short_text = text[:avail_for_text]
        new_content = f"  {short_text}{ell}  "
        # 如果 new_content 恰好等于 length，直接打印；如仍不等，可对齐处理
        if len(new_content) == length:
            print(new_content)
            return
        # 若有微小差异（通常不会），用填充使其精确到 length
        total_pad = length - len(new_content)
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        left = make_pad(left_pad)
        right = make_pad(right_pad)
        print(f"{left}{new_content}{right}")
        return

    # avail_for_text <= 0：无法在保留左右两个空格与整 '...' 的前提下放入任何文字
    # 退而求其次：如果 length >= left_right_spaces + 1，则放置一个省略号（或其部分）
    min_needed = left_right_spaces + 1  # 至少留一个可见字符
    if length >= (left_right_spaces + 1):
        # 我们尽量打印 "  ...  " 的一部分或全部
        # 先尝试完整 ell，如果空间不够则截取 ell 的前缀
        ell_part_len = max(1, length - left_right_spaces)
        ell_part = ell[:ell_part_len]
        out = f"  {ell_part}  "
        # 若 out 长度超出（极端情况），直接截断到 length
        print(out[:length])
        return

    # 最极端：length 非常小，无法满足两侧空格要求，直接打印被截短的 ell 的前缀或空字符串
    print(ell[:max(0, length)])


def print_dict_values(
    d: Dict[Any, Any],
    key: Any,
    pre_str: Optional[Union[str, Sequence[str]]] = None,
    start_index: int = 0
) -> None:
    """
    打印字典 d 中 key 对应的值（逐元素逐行打印）。
    - 若 key 是单个 key（非序列或字符串），行为与以前一致：
        * 若对应值为序列（排除 str/bytes），逐行打印每个元素（索引从 start_index 开始或使用 pre_str）
        * 若对应值为单值，直接打印该值
    - 若 key 是序列（如 list/tuple），则把序列中的每个元素视作独立的字典 key：
        * 按行输出每个索引对应的值，格式 "label: v1, v2, v3"，
          其中 v1 对应第一个 key 的第 i 个元素，v2 对应第二个 key 的第 i 个元素，依此类推。
        * 对于某个 key 缺失，使用 "(missing)" 占位；对于序列长度不足的索引，使用 ""（空字符串）占位。
        * 如果所有 key 的值都是标量（非序列），则只输出一行，显示这些标量。
    - pre_str:
        * None（默认）：使用序号 label（从 start_index 开始）
        * str：使用 f"{pre_str}{index}" 作为 label
        * 若为序列（且非 str），且长度等于输出行数，则用该序列元素作为每行的 label；否则回退为序号并给出提示

    v2:
        Robust version that safely handles numpy arrays, sequences, scalars, sets, missing keys, and multi-key lists.
        See previous docstrings for behavior; this version fixes ambiguous truth-value errors and formats array-like values.
    """
    def _is_seq(obj):
        # treat numpy arrays as sequence-like too, but exclude str/bytes
        return (isinstance(obj, _Sequence) and not isinstance(obj, (str, bytes))) or isinstance(obj, np.ndarray)

    def _to_items(val):
        # Normalize val into a list for indexable access in multi-key scenario
        if val is None:
            return [None]
        if isinstance(val, set):
            return list(val)
        if isinstance(val, np.ndarray):
            # numpy scalar -> single-item list
            if val.ndim == 0:
                return [val.item()]
            # convert ndarray to python list (handles nested arrays)
            return list(val.tolist())
        if _is_seq(val):
            return list(val)
        # scalar -> single-element list (will be repeated across rows)
        return [val]

    def _fmt(v):
        """Format any supported value into a single string safely."""
        if v is None:
            return "(missing)"
        # strings and bytes: keep as-is
        if isinstance(v, (str, bytes)):
            return v
        # numpy scalar (np.int64, np.float64, etc.)
        if isinstance(v, np.generic):
            try:
                return str(v.item())
            except Exception:
                return str(v)
        # numpy arrays
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return "[]"
            if v.ndim == 0:
                return str(v.item())
            # try to convert to list and format elements
            try:
                lst = v.tolist()
                # lst may be nested; recursively format elements
                return "[" + ", ".join(_fmt(x) for x in lst) + "]"
            except Exception:
                return np.array2string(v, threshold=10)
        # other sequences (list/tuple/... but not str/bytes)
        if isinstance(v, _Sequence) and not isinstance(v, (str, bytes)):
            try:
                return "[" + ", ".join(_fmt(x) for x in list(v)) + "]"
            except Exception:
                return str(v)
        # fallback scalar (int/float/bool/etc.)
        return str(v)

    # ---- begin main logic ----

    # multi-key (key is a sequence but not str/bytes)
    if _is_seq(key) and not isinstance(key, (str, bytes)):
        keys = list(key)
        if len(keys) == 0:
            print("[Warning] provided empty key list.")
            return

        # collect items lists for each requested key
        items_per_key = []
        for k in keys:
            if k not in d:
                # use explicit None placeholder to indicate missing key
                print(f"[Warning] key {repr(k)} not found in dictionary; using (missing) placeholder.")
                items_per_key.append([None])
            else:
                items_per_key.append(_to_items(d[k]))

        # number of rows = max length among the key-value lists (scalars become length 1)
        max_rows = max((len(lst) for lst in items_per_key), default=0)
        if max_rows == 0:
            print("[Info] all selected keys map to empty sequences.")
            return

        # label handling
        label_mode = "index"
        use_labels = None
        if pre_str is None:
            label_mode = "index"
        elif isinstance(pre_str, str):
            label_mode = "prefix_str"
        elif _is_seq(pre_str) and not isinstance(pre_str, (str, bytes)):
            try:
                if len(pre_str) == max_rows:
                    use_labels = list(pre_str)
                    label_mode = "seq"
                else:
                    print("[Note] pre_str is a sequence but length != number of rows; fallback to index labels.")
                    label_mode = "index"
            except TypeError:
                print("[Note] pre_str provided but not usable for element-wise labeling; fallback to index labels.")
                label_mode = "index"
        else:
            label_mode = "index"

        # print rows
        for i in range(max_rows):
            row_vals = []
            for lst in items_per_key:
                if i < len(lst):
                    v = lst[i]
                    row_vals.append(_fmt(v))
                else:
                    # out-of-range for this key -> empty placeholder (keeps CSV alignment)
                    row_vals.append("")
            joined = ", ".join(row_vals)
            if label_mode == "index":
                label = f"{start_index + i}"
            elif label_mode == "prefix_str":
                label = f"{pre_str}{start_index + i}"
            else:  # seq labels
                label = str(use_labels[i])
            print(f"{label}: {joined}")
        return

    # single-key behavior
    single_key = key
    if single_key not in d:
        print(f"[Warning] key {repr(single_key)} not found in dictionary.")
        return

    val = d[single_key]

    # if scalar-like (not sequence and not numpy array) -> print as single value
    if not _is_seq(val) and not isinstance(val, set):
        print(_fmt(val))
        return

    # convert set/sequence to list and print per-element lines
    if isinstance(val, set):
        items = list(val)
    elif isinstance(val, np.ndarray):
        # ndarray -> list (ndarray ->tolist may produce nested lists)
        if val.ndim == 0:
            items = [val.item()]
        else:
            items = list(val.tolist())
    else:
        items = list(val)

    if len(items) == 0:
        print(f"[Info] {repr(single_key)} -> (empty)")
        return

    # pre_str handling for single key
    if pre_str is None:
        for i, v in enumerate(items):
            print(f"{start_index + i}: {_fmt(v)}")
        return

    if isinstance(pre_str, str):
        for i, v in enumerate(items):
            print(f"{pre_str}{start_index + i}: {_fmt(v)}")
        return

    if _is_seq(pre_str) and not isinstance(pre_str, (str, bytes)):
        try:
            if len(pre_str) == len(items):
                for i, v in enumerate(items):
                    print(f"{pre_str[i]}: {_fmt(v)}")
                return
            else:
                print("[Note] pre_str is a sequence but length != number of values; fallback to indices.")
        except TypeError:
            print("[Note] pre_str provided but not usable for element-wise labeling; fallback to indices.")

    for i, v in enumerate(items):
        print(f"{start_index + i}: {_fmt(v)}")
