import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S',
                    filename=r'D:\Documents\Coding\Python\SEUCAM\Channel Measurement\logging\decode.log',
                    filemode='w')


def decode_bytes(byte_data, pth,*,project_pth):
    # ---------- 文件头解析 ----------
    all_bytes = byte_data.tobytes()
    parts = all_bytes.split(b'\x00')
    filename = parts[0].decode(errors='ignore')
    filesize = int(parts[1].decode(errors='ignore'))
    header_length = len(parts[0]) + 1 + len(parts[1]) + 1

    file_bytes = byte_data[header_length: header_length + filesize]

    if not os.path.exists(pth):
        os.makedirs(pth)

    with open(os.path.join(pth, filename), "wb") as f:
        f.write(file_bytes)

    logging.info(f"Decoding done, result saved in： {os.path.join(pth, filename)}")